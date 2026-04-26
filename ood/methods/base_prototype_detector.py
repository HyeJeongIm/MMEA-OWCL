"""
Base class for Prototype Penalty Score OOD detectors.

All distance-metric variants (Cosine, kNN, DiagMahal, SoftmaxL2, PooledMahal)
inherit from this class and implement _compute_dist_for_modality(z, m).

Key design:
  - dist_stats self-computed from raw_logit_arrays using subclass distance metric
    → fixes the L2/Cosine stats mismatch in earlier implementations
  - gamma=0 → no KL penalty (base variant)
  - gamma>0 → cross-modal KL penalty added (CrossModal variant)
"""

import logging
import numpy as np
import torch

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class BasePrototypePenaltyDetector:
    """
    Args:
        prototypes:        {m: {class_idx: np.ndarray [C]}}  mean prototype
        raw_logit_arrays:  {m: {class_idx: list[np.ndarray [C]]}}  all buffer vecs
        modality:          ordered auxiliary modality list e.g. ['RGB','Gyro','Acce']
        device:            torch device string
        beta:              prototype distance penalty weight
        gamma:             cross-modal KL penalty weight  (0 = no KL term)
        alpha_temp:        softmax temperature for modality reliability weights
        dist_stats:        fallback if raw_logit_arrays is empty
    """

    def __init__(self, prototypes: dict, modality: list,
                 device: str = "cuda", beta: float = 1.0,
                 gamma: float = 0.0, alpha_temp: float = 1.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None):
        self.prototypes = prototypes
        self.raw_logit_arrays = raw_logit_arrays or {}
        self.modality = modality
        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.alpha_temp = alpha_temp
        self._wandb_logged = False

        self._setup()

        if self.raw_logit_arrays:
            self.dist_stats = self._fit_dist_stats()
        else:
            self.dist_stats = dist_stats or {}

        logging.info(
            "[%s] β=%.2f, γ=%.2f, T=%.2f  dist_stats_keys=%s",
            self.__class__.__name__, beta, gamma, alpha_temp,
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
        """Compute (μ, σ) per modality using this detector's own distance metric."""
        dist_stats = {}
        for m, cls_dict in self.raw_logit_arrays.items():
            if m not in self.prototypes or not cls_dict:
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

    def compute_scores(self, z_main, aux_logits: dict) -> np.ndarray:
        z_main_np = self._to_np(z_main)

        avail = [
            m for m in self.modality
            if m in aux_logits
            and m in self.prototypes and len(self.prototypes[m]) > 0
            and m in self.dist_stats
        ]

        if not avail:
            logging.warning(
                "[%s] No valid modalities — fallback: max(z_main)",
                self.__class__.__name__,
            )
            return z_main_np.max(axis=1)

        z_aux = {m: self._to_np(aux_logits[m]) for m in avail}

        # Step 1: distance → normalized score u_proto, penalty term ood_m
        u_proto, ood_m = {}, {}
        for m in avail:
            raw_dist = self._compute_dist_for_modality(z_aux[m], m)  # [N]
            mu_d, sig_d = self.dist_stats[m]
            normalized = (raw_dist - mu_d) / (sig_d + 1e-8)
            u_proto[m] = -normalized
            ood_m[m] = np.maximum(0.0, normalized)

        # Step 2: α weights → z_final
        u_stack = np.stack([u_proto[m] for m in avail], axis=1)  # [N, M]
        u_stack -= u_stack.max(axis=1, keepdims=True)
        exp_u = np.exp(u_stack / self.alpha_temp)
        alpha = exp_u / exp_u.sum(axis=1, keepdims=True)           # [N, M]

        z_final = z_main_np.copy()
        for k, m in enumerate(avail):
            z_final = z_final + alpha[:, k:k+1] * z_aux[m]

        # Step 3a: prototype distance penalty
        P_proto = np.stack([ood_m[m] for m in avail], axis=1).mean(axis=1)  # [N]

        # Step 3b: cross-modal KL penalty (only when gamma > 0)
        if self.gamma > 0:
            all_logits = {"main": z_main_np}
            all_logits.update(z_aux)
            p_all = [self._softmax_np(all_logits[k]) for k in ["main"] + avail]
            p_bar = np.mean(np.stack(p_all, axis=0), axis=0)
            D_kl = np.mean(
                np.stack([self._kl_div(p, p_bar) for p in p_all], axis=0), axis=0
            )
            scores = z_final.max(axis=1) - self.beta * P_proto - self.gamma * D_kl
        else:
            D_kl = np.zeros(len(z_main_np))
            scores = z_final.max(axis=1) - self.beta * P_proto

        self._log_stats(avail, alpha, P_proto, D_kl)
        return scores

    def _log_stats(self, avail, alpha, P_proto, D_kl) -> None:
        w_mean = alpha.mean(axis=0)
        name = self.__class__.__name__
        logging.info(
            "[%s] α — %s | P_proto=%.4f, D_kl=%.4f", name,
            ", ".join(f"{m}: {w_mean[k]:.4f}" for k, m in enumerate(avail)),
            float(P_proto.mean()), float(D_kl.mean()),
        )
        if _WANDB_AVAILABLE and not self._wandb_logged:
            log_dict = {
                f"{name}/P_proto_mean": float(P_proto.mean()),
                f"{name}/D_kl_mean":   float(D_kl.mean()),
            }
            for k, m in enumerate(avail):
                log_dict[f"{name}/alpha_mean_{m}"] = float(w_mean[k])
            try:
                _wandb.log(log_dict)
                self._wandb_logged = True
            except Exception as e:
                logging.debug("[%s] wandb.log failed: %s", name, e)

    def compute_scores_from_outputs(self, outputs: dict) -> np.ndarray:
        return self.compute_scores(
            outputs["logits"],
            outputs.get("auxiliary_logits", {}),
        )
