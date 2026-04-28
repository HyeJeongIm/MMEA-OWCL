"""
MoASBase — Abstract base class for MoAS (Modality-adaptive OOD Scoring).

Subclasses implement _compute_dist_for_modality() to provide the distance
metric (e.g. 1-NN L2 in KNNMoASDetector). The scoring algorithm (Eq. 2–7)
is fully implemented here.

Paper notation (ICMLW):
  d̃_m  — normalised kNN distance per modality               [Eq. 2]
  α_m   — modality reliability weight = softmax(-d̃_m / τ)   [Eq. 3]
  τ     (tau)   — temperature for α_m weights                [Eq. 3]
  η     (eta)   — known-class deviation penalty weight        [Eq. 7]
  γ     (gamma) — cross-modal KL disagreement penalty weight  [Eq. 7]
  P(x)  — known-class deviation penalty = mean_m max(0, d̃_m) [Eq. 5]
  D_KL  — cross-modal KL disagreement                        [Eq. 6]

Score decomposition for component analysis:
  A(x) = max_c z_final,c(x)   — adaptive fusion contribution
  B(x) = η · P(x)              — deviation penalty
  C(x) = γ · D_KL(x)           — KL disagreement penalty
  s(x) = A(x) - B(x) - C(x)   — final MoAS score  [Eq. 7]
"""

import logging
import numpy as np
import torch

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class MoASBase:
    """
    Args:
        raw_logit_arrays:  {m: {class_idx: list[np.ndarray [C]]}}  buffer vecs (kNN dist_ref 구성)
        modality:          ordered auxiliary modality list e.g. ['RGB','Gyro','Acce']
        device:            torch device string
        eta:               known-class deviation penalty weight η  [Eq. 7]
        gamma:             cross-modal KL disagreement penalty weight γ  [Eq. 7]
                           (0 = no KL term)
        tau:               softmax temperature τ for modality reliability weights α_m  [Eq. 3]
        dist_stats:        precomputed (μ_m, σ_m) fallback when raw_logit_arrays is empty
    """

    def __init__(self, modality: list,
                 device: str = "cuda", eta: float = 1.0,
                 gamma: float = 0.0, tau: float = 1.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None,
                 **kwargs):
        self.raw_logit_arrays = raw_logit_arrays or {}
        self.modality = modality
        self.device = device
        self.eta = eta
        self.gamma = gamma
        self.tau = tau
        self._wandb_logged = False

        self._setup()

        if self.raw_logit_arrays:
            self.dist_stats = self._fit_dist_stats()
        else:
            self.dist_stats = dist_stats or {}

        logging.info(
            "[%s] η=%.2f, γ=%.2f, τ=%.2f  dist_stats_keys=%s",
            self.__class__.__name__, eta, gamma, tau,
            list(self.dist_stats.keys()),
        )

    def _setup(self):
        """Override to pre-compute auxiliary data before _fit_dist_stats is called."""
        pass

    def _compute_dist_for_modality(self, z: np.ndarray, m: str) -> np.ndarray:
        """
        z: [N, C] logit array for modality m.
        Returns [N] distances to nearest prototype.
        Must be implemented by subclass.
        """
        raise NotImplementedError

    def _fit_dist_stats(self) -> dict:
        """Compute (μ_m, σ_m) per modality using this detector's own distance metric."""
        dist_stats = {}
        for m, cls_dict in self.raw_logit_arrays.items():
            if not cls_dict:
                continue
            dists = []
            for c, vecs in cls_dict.items():
                if not vecs:
                    continue
                z = np.stack(vecs, axis=0)      # [K, C]
                d = self._compute_dist_for_modality(z, m)  # [K]
                dists.extend(d.tolist())
            if dists:
                arr = np.array(dists)
                dist_stats[m] = (float(arr.mean()), float(arr.std()) + 1e-8)
                logging.info(
                    "[%s] %s: μ=%.4f, σ=%.4f (N=%d)",
                    self.__class__.__name__, m,
                    dist_stats[m][0], dist_stats[m][1], len(dists),
                )
        return dist_stats

    @staticmethod
    def _to_np(x) -> np.ndarray:
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    @staticmethod
    def _softmax_np(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        return (p * np.log(p / q)).sum(axis=1)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_all(self, z_main, aux_logits: dict) -> dict:
        """
        Full MoAS computation. Returns all components for analysis.

        Returns:
            dict with keys:
              'scores' [N]    — s(x) = A(x) - B(x) - C(x)
              'A'      [N]    — max_c z_final,c(x)    (adaptive fusion term)
              'B'      [N]    — η · P(x)               (deviation penalty)
              'C'      [N]    — γ · D_KL(x)            (KL disagreement)
              'alpha'  [N,M]  — α_m per sample         (modality weights)
              'avail'  list   — modalities that were used
        """
        z_main_np = self._to_np(z_main)

        avail = [
            m for m in self.modality
            if m in aux_logits
            and m in self.dist_stats
        ]

        if not avail:
            logging.warning(
                "[%s] No valid modalities — fallback: max(z_main)",
                self.__class__.__name__,
            )
            zeros = np.zeros(len(z_main_np))
            A = z_main_np.max(axis=1)
            return {
                'scores': A.copy(),
                'A': A,
                'B': zeros,
                'C': zeros,
                'alpha': None,
                'avail': [],
            }

        z_aux = {m: self._to_np(aux_logits[m]) for m in avail}

        # Step 1: d̃_m = (d_m - μ_m) / σ_m  [Eq. 2]
        #   d_tilde[m]    : [N]  normalised distance  (core discriminative signal)
        #   neg_d_tilde[m]: [N]  -d̃_m  (softmax input for α_m)
        #   pos_d_tilde[m]: [N]  max(0, d̃_m)  (contribution to P(x))
        d_tilde, neg_d_tilde, pos_d_tilde = {}, {}, {}
        for m in avail:
            raw_dist = self._compute_dist_for_modality(z_aux[m], m)  # [N]
            mu_d, sig_d = self.dist_stats[m]
            d_tilde[m]     = (raw_dist - mu_d) / (sig_d + 1e-8)      # d̃_m
            neg_d_tilde[m] = -d_tilde[m]
            pos_d_tilde[m] = np.maximum(0.0, d_tilde[m])

        # Step 2: α_m = softmax(-d̃_m / τ)  [Eq. 3]
        #         z_final = z_main + Σ_m α_m · z_m  [Eq. 4]
        u_stack = np.stack([neg_d_tilde[m] for m in avail], axis=1)  # [N, M]
        u_stack -= u_stack.max(axis=1, keepdims=True)
        exp_u = np.exp(u_stack / self.tau)
        alpha = exp_u / exp_u.sum(axis=1, keepdims=True)             # [N, M]

        z_final = z_main_np.copy()
        for k, m in enumerate(avail):
            z_final = z_final + alpha[:, k:k+1] * z_aux[m]

        # Step 3a: P(x) = mean_m max(0, d̃_m)  [Eq. 5]
        P_x = np.stack([pos_d_tilde[m] for m in avail], axis=1).mean(axis=1)  # [N]

        # Step 3b: D_KL(x)  [Eq. 6]  (only when γ > 0)
        if self.gamma > 0:
            all_logits = {"main": z_main_np}
            all_logits.update(z_aux)
            p_all = [self._softmax_np(all_logits[k]) for k in ["main"] + avail]
            p_bar = np.mean(np.stack(p_all, axis=0), axis=0)
            D_KL = np.mean(
                np.stack([self._kl_div(p, p_bar) for p in p_all], axis=0), axis=0
            )
        else:
            D_KL = np.zeros(len(z_main_np))

        # A(x), B(x), C(x), s(x)  [Eq. 7]
        A = z_final.max(axis=1)
        B = self.eta * P_x
        C = self.gamma * D_KL
        scores = A - B - C

        # d̃_m stacked: [N, M]  (row = sample, col = modality)
        d_tilde_stack = np.stack([d_tilde[m] for m in avail], axis=1)

        self._log_stats(avail, alpha, P_x, D_KL)
        return {
            'scores': scores,           # [N]    s(x) = A - B - C
            'A': A,                     # [N]    max z_final  (fusion term)
            'B': B,                     # [N]    η·P(x)       (deviation penalty)
            'C': C,                     # [N]    γ·D_KL(x)    (KL penalty)
            'P_x': P_x,                 # [N]    raw deviation penalty (before η)
            'D_KL': D_KL,              # [N]    raw KL divergence (before γ)
            'alpha': alpha,             # [N, M] α_m per sample
            'd_tilde': d_tilde_stack,   # [N, M] d̃_m per sample (normalised kNN dist)
            'avail': avail,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_scores(self, z_main, aux_logits: dict) -> np.ndarray:
        """Returns [N] MoAS scores s(x) = A(x) - B(x) - C(x)."""
        return self._compute_all(z_main, aux_logits)['scores']

    def compute_scores_detailed(self, z_main, aux_logits: dict) -> dict:
        """
        Returns all MoAS components for analysis:
          A(x), B(x), C(x), scores, alpha, avail
        See _compute_all() for full documentation.
        """
        return self._compute_all(z_main, aux_logits)

    def compute_scores_from_outputs(self, outputs: dict) -> np.ndarray:
        """compute_scores wrapper accepting the outputs dict."""
        return self.compute_scores(
            outputs["logits"],
            outputs.get("auxiliary_logits", {}),
        )

    def compute_scores_detailed_from_outputs(self, outputs: dict) -> dict:
        """compute_scores_detailed wrapper accepting the outputs dict."""
        return self.compute_scores_detailed(
            outputs["logits"],
            outputs.get("auxiliary_logits", {}),
        )

    # ------------------------------------------------------------------
    # Internal logging
    # ------------------------------------------------------------------

    def _log_stats(self, avail, alpha, P_x, D_KL) -> None:
        w_mean = alpha.mean(axis=0)
        name = self.__class__.__name__
        logging.info(
            "[%s] α — %s | P(x)=%.4f, D_KL=%.4f", name,
            ", ".join(f"{m}: {w_mean[k]:.4f}" for k, m in enumerate(avail)),
            float(P_x.mean()), float(D_KL.mean()),
        )
        if _WANDB_AVAILABLE and not self._wandb_logged:
            log_dict = {
                f"{name}/P_x_mean":   float(P_x.mean()),
                f"{name}/D_KL_mean":  float(D_KL.mean()),
            }
            for k, m in enumerate(avail):
                log_dict[f"{name}/alpha_mean_{m}"] = float(w_mean[k])
            try:
                _wandb.log(log_dict)
                self._wandb_logged = True
            except Exception as e:
                logging.debug("[%s] wandb.log failed: %s", name, e)
