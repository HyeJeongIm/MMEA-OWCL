"""
MoAS detectors — kNN-based Modality-adaptive OOD Scoring.

dist_ref_m = concat([buffer_vecs_c for all c])     [N_total, C]
d_m       = min_i ||z_m - dist_ref_m[i]||_2        (1-NN L2 distance)

Class hierarchy:
  MoASBase                (moas_base.py)  — MoAS scoring algorithm  [Eq. 2–7]
    └─ KNNMoASDetector                   — 1-NN L2 distance metric
         └─ CrossModalKNNMoASDetector    — γ > 0 default (KL term enabled)
              └─ MoASDetector            — paper defaults: τ=3, η=3, γ=4

Ablation variants (γ=0 or η=0):
  MoASAdaptiveFusionOnly    : score = max(z_fused)                       (η=γ=0)
  MoASDeviationPenalty      : score = max(z_fused) - η·P(x)             (γ=0)
  MoASKLPenalty             : score = max(z_fused) - γ·D_KL(x)          (η=0)
  MoASDetector (full MoAS)  : score = max(z_fused) - η·P(x) - γ·D_KL(x)  [Eq. 7]

Paper notation:
  τ (tau)   — temperature for α_m weights   [Eq. 3]
  η (eta)   — deviation penalty weight       [Eq. 7]
  γ (gamma) — KL disagreement penalty weight [Eq. 7]
"""

import logging
import numpy as np
from ood.methods.moas_base import MoASBase


class KNNMoASDetector(MoASBase):
    """MoAS with 1-NN L2 distance metric.

    Implements _compute_dist_for_modality() using a per-modality dist_ref
    built from all buffer samples (leave-one-out statistics).

    Args:
        raw_logit_arrays:  {m: {class_idx: list[np.ndarray [C]]}}  buffer vecs for dist_ref
        modality:          ordered auxiliary modality list
        device:            torch device string
        eta:               known-class deviation penalty weight η  (default 1.0)
        gamma:             cross-modal KL disagreement penalty weight γ  (default 0.0)
        tau:               softmax temperature τ for α_m weights  (default 1.0)
        dist_stats:        precomputed (μ_m, σ_m) fallback when raw_logit_arrays is empty
    """

    def __init__(self, modality: list,
                 device: str = "cuda", eta: float = 1.0, gamma: float = 0.0,
                 tau: float = 1.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            modality=modality,
            device=device,
            eta=eta,
            gamma=gamma,
            tau=tau,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )

    def _setup(self):
        """Build per-modality dist_ref from all buffer samples."""
        self._dist_ref = {}
        for m, cls_dict in self.raw_logit_arrays.items():
            all_vecs = [
                np.stack(vecs, axis=0)
                for vecs in cls_dict.values()
                if len(vecs) > 0
            ]
            if all_vecs:
                self._dist_ref[m] = np.concatenate(all_vecs, axis=0)  # [N_total, C]
                logging.info(
                    "[%s] dist_ref[%s]: %d vectors",
                    self.__class__.__name__, m, len(self._dist_ref[m]),
                )

    def _fit_dist_stats(self) -> dict:
        """Leave-one-out 1-NN statistics (μ_m, σ_m) to avoid μ=0 from self-hit."""
        dist_stats = {}
        for m, cls_dict in self.raw_logit_arrays.items():
            if not cls_dict:
                continue
            dist_ref = self._dist_ref.get(m)
            if dist_ref is None or len(dist_ref) < 2:
                continue
            all_vecs = np.concatenate(
                [np.stack(vecs, axis=0) for vecs in cls_dict.values() if vecs], axis=0
            )  # [N_total, C]
            dists = []
            for i, z in enumerate(all_vecs):
                loo = np.delete(dist_ref, i, axis=0)  # [N_total-1, C]  exclude self
                d = float(np.linalg.norm(z[None, :] - loo, axis=1).min())
                dists.append(d)
            arr = np.array(dists)
            dist_stats[m] = (float(arr.mean()), float(arr.std()) + 1e-8)
            logging.info(
                "[%s] %s: μ=%.4f, σ=%.4f (N=%d, leave-one-out)",
                self.__class__.__name__, m, dist_stats[m][0], dist_stats[m][1], len(dists),
            )
        return dist_stats

    def _compute_dist_for_modality(self, z: np.ndarray, m: str) -> np.ndarray:
        """z: [N, C] → [N] 1-NN L2 distance to buffer dist_ref."""
        dist_ref = self._dist_ref.get(m)
        if dist_ref is None or len(dist_ref) == 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] dist_ref for modality '{m}' is empty. "
                "raw_logit_arrays must be provided to build the dist_ref."
            )
        diff = z[:, None, :] - dist_ref[None, :, :]                # [N, N_total, C]
        return np.linalg.norm(diff, axis=-1).min(axis=1)          # [N]


class CrossModalKNNMoASDetector(KNNMoASDetector):
    """KNNMoASDetector with cross-modal KL disagreement penalty enabled (γ > 0)."""

    def __init__(self, modality: list,
                 device: str = "cuda", eta: float = 1.0, gamma: float = 0.5,
                 tau: float = 1.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            modality=modality,
            device=device,
            eta=eta,
            gamma=gamma,
            tau=tau,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )


# =============================================================================
# MoAS variants (full + ablation)
# =============================================================================

class MoASDetector(CrossModalKNNMoASDetector):
    """Full MoAS: adaptive fusion + deviation penalty + KL disagreement.  [Eq. 7]

    score = max(z_fused) - η·P(x) - γ·D_KL(x)

    Default hyperparameters (Appendix B.1):  τ=3, η=3, γ=4
    """

    def __init__(self, modality: list,
                 device: str = "cuda", eta: float = 3.0, gamma: float = 4.0,
                 tau: float = 3.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            modality=modality,
            device=device,
            eta=eta,
            gamma=gamma,
            tau=tau,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )


class MoASAdaptiveFusionOnly(KNNMoASDetector):
    """MoAS ablation — adaptive fusion only, no penalty terms.

    score = max(z_fused)

    Measures the contribution of modality-reliability-weighted fusion (α_m)
    alone, without any penalty. η=γ=0 regardless of YAML values; only τ applies.
    """

    def __init__(self, modality: list,
                 device: str = "cuda", tau: float = 3.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            modality=modality,
            device=device,
            eta=0.0,    # no deviation penalty
            gamma=0.0,  # no KL penalty
            tau=tau,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )


class MoASDeviationPenalty(KNNMoASDetector):
    """MoAS ablation — adaptive fusion + known-class deviation penalty only.

    score = max(z_fused) - η·P(x)

    Measures the contribution of P(x) in isolation. γ=0 always.
    """

    def __init__(self, modality: list,
                 device: str = "cuda", eta: float = 3.0, tau: float = 3.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            modality=modality,
            device=device,
            eta=eta,
            gamma=0.0,  # no KL penalty
            tau=tau,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )


class MoASKLPenalty(KNNMoASDetector):
    """MoAS ablation — adaptive fusion + cross-modal KL penalty only.

    score = max(z_fused) - γ·D_KL(x)

    Measures the contribution of D_KL in isolation. η=0 always.
    """

    def __init__(self, modality: list,
                 device: str = "cuda", gamma: float = 4.0, tau: float = 3.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            modality=modality,
            device=device,
            eta=0.0,    # no deviation penalty
            gamma=gamma,
            tau=tau,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (deprecated names → new names)
# ---------------------------------------------------------------------------
KNNPrototypePenaltyScoreDetector           = KNNMoASDetector
CrossModalKNNPrototypePenaltyScoreDetector = CrossModalKNNMoASDetector
MoASConfidenceOnlyDetector                 = MoASAdaptiveFusionOnly
MoASDistancePenaltyDetector                = MoASDeviationPenalty
MoASKLPenaltyDetector                      = MoASKLPenalty
