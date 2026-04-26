"""
kNN Prototype Penalty Score — Post-hoc OOD Detector.

PrototypePenaltyScore의 "nearest class mean" distance를
"nearest buffer sample" (1-NN) distance로 교체.

동기:
  class mean prototype은 클래스 내 분산을 무시함.
  버퍼 10개/class를 모두 gallery로 사용하면 더 세밀한 거리 측정이 가능.
  특히 bimodal한 클래스 분포에서 mean이 density가 낮은 곳에 위치할 때 유리.

수식:
  gallery_m = concat([buffer_vecs_c for all c])   [N_total, C]
  raw_dist  = min_i ||z_m - gallery_m[i]||_2      (1-NN)

두 가지 base variant:
  KNNPrototypePenaltyScore              : score = max(z_final) - β·P
  CrossModalKNNPrototypePenaltyScore    : score = max(z_final) - β·P - γ·D_kl

MoAS ablation variants (논문 ablation study용):
  MoAS_ConfidenceOnly    : score = max(z_fused)               (adaptive fusion only)
  MoAS_DistancePenalty   : score = max(z_fused) - α·P         (+ distance penalty)
  MoAS_KLPenalty         : score = max(z_fused) - γ·D_kl      (+ KL penalty)
  MoAS                   : score = max(z_fused) - α·P - γ·D_kl (full, = CrossModalKNN)
"""

import logging
import numpy as np
from ood.methods.base_prototype_detector import BasePrototypePenaltyDetector


class KNNPrototypePenaltyScoreDetector(BasePrototypePenaltyDetector):
    """
    Args:
        prototypes:        {m: {class_idx: np.ndarray [C]}}  (α 계산에만 사용)
        raw_logit_arrays:  {m: {class_idx: list[np.ndarray [C]]}}  gallery 구성에 사용
        modality:          ordered auxiliary modality list
        device:            torch device string
        beta:              prototype distance penalty weight (default 1.0)
        gamma:             cross-modal KL penalty weight (default 0.0)
        alpha_temp:        softmax temperature (default 1.0)
        dist_stats:        fallback if raw_logit_arrays is empty
    """

    def __init__(self, prototypes: dict, modality: list,
                 device: str = "cuda", beta: float = 1.0, gamma: float = 0.0,
                 alpha_temp: float = 1.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            prototypes=prototypes,
            modality=modality,
            device=device,
            beta=beta,
            gamma=gamma,
            alpha_temp=alpha_temp,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )

    def _setup(self):
        """Build per-modality gallery from all buffer samples."""
        self._gallery = {}
        for m, cls_dict in self.raw_logit_arrays.items():
            all_vecs = [
                np.stack(vecs, axis=0)
                for vecs in cls_dict.values()
                if len(vecs) > 0
            ]
            if all_vecs:
                self._gallery[m] = np.concatenate(all_vecs, axis=0)  # [N_total, C]
                logging.info(
                    "[%s] gallery[%s]: %d vectors",
                    self.__class__.__name__, m, len(self._gallery[m]),
                )

    def _fit_dist_stats(self) -> dict:
        """Leave-one-out kNN dist_stats to avoid μ=0 from self-hit."""
        dist_stats = {}
        for m, cls_dict in self.raw_logit_arrays.items():
            if m not in self.prototypes or not cls_dict:
                continue
            gallery = self._gallery.get(m)
            if gallery is None or len(gallery) < 2:
                continue
            all_vecs = np.concatenate(
                [np.stack(vecs, axis=0) for vecs in cls_dict.values() if vecs], axis=0
            )  # [N_total, C]
            dists = []
            for i, z in enumerate(all_vecs):
                # exclude self from gallery (leave-one-out)
                loo = np.delete(gallery, i, axis=0)  # [N_total-1, C]
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
        """z: [N, C] → [N] 1-NN distance to buffer gallery."""
        gallery = self._gallery.get(m)
        if gallery is None or len(gallery) == 0:
            # fallback to nearest prototype L2 distance
            protos = np.array(list(self.prototypes[m].values()))  # [K, C]
            diff = z[:, None, :] - protos[None, :, :]            # [N, K, C]
            return np.linalg.norm(diff, axis=-1).min(axis=1)

        diff = z[:, None, :] - gallery[None, :, :]               # [N, N_total, C]
        return np.linalg.norm(diff, axis=-1).min(axis=1)         # [N]


class CrossModalKNNPrototypePenaltyScoreDetector(KNNPrototypePenaltyScoreDetector):
    """kNN distance + cross-modal KL disagreement penalty (γ=0.5)."""

    def __init__(self, prototypes: dict, modality: list,
                 device: str = "cuda", beta: float = 1.0, gamma: float = 0.5,
                 alpha_temp: float = 1.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            prototypes=prototypes,
            modality=modality,
            device=device,
            beta=beta,
            gamma=gamma,
            alpha_temp=alpha_temp,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )


# =============================================================================
# MoAS ablation variants
# =============================================================================

class MoASDetector(CrossModalKNNPrototypePenaltyScoreDetector):
    """Full MoAS: adaptive fusion + distance penalty + KL disagreement.

    Equivalent to CrossModalKNNPrototypePenaltyScore.
    score = max(z_fused) - α·P(x) - γ·D_KL(x)
    """

    def __init__(self, prototypes: dict, modality: list,
                 device: str = "cuda", beta: float = 3.0, gamma: float = 4.0,
                 alpha_temp: float = 3.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            prototypes=prototypes,
            modality=modality,
            device=device,
            beta=beta,
            gamma=gamma,
            alpha_temp=alpha_temp,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )


class MoASConfidenceOnlyDetector(KNNPrototypePenaltyScoreDetector):
    """MoAS ablation — confidence only: adaptive fusion, no penalty terms.

    Uses modality-reliability weights w_m (kNN-based) to fuse z_main and z_m,
    but applies no prototype distance penalty and no KL disagreement penalty.
    score = max(z_fused)

    핵심 주장: adaptive fusion 자체의 기여를 단독으로 측정.
    beta와 gamma는 항상 0 (YAML 값 무관). alpha_temp만 YAML에서 사용.
    """

    def __init__(self, prototypes: dict, modality: list,
                 device: str = "cuda", alpha_temp: float = 3.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            prototypes=prototypes,
            modality=modality,
            device=device,
            beta=0.0,   # always 0: no distance penalty
            gamma=0.0,  # always 0: no KL penalty
            alpha_temp=alpha_temp,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )


class MoASDistancePenaltyDetector(KNNPrototypePenaltyScoreDetector):
    """MoAS ablation — confidence + distance penalty only (no KL term).

    score = max(z_fused) - α·P(x)

    핵심 주장: prototype distance penalty만의 기여를 측정.
    beta(α)는 YAML ood_beta 사용. gamma는 항상 0.
    """

    def __init__(self, prototypes: dict, modality: list,
                 device: str = "cuda", beta: float = 3.0, alpha_temp: float = 3.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            prototypes=prototypes,
            modality=modality,
            device=device,
            beta=beta,
            gamma=0.0,  # always 0: no KL penalty
            alpha_temp=alpha_temp,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )


class MoASKLPenaltyDetector(KNNPrototypePenaltyScoreDetector):
    """MoAS ablation — confidence + KL disagreement penalty only (no distance term).

    score = max(z_fused) - γ·D_KL(x)

    핵심 주장: cross-modal KL disagreement만의 기여를 측정.
    gamma(γ)는 YAML ood_gamma 사용. beta는 항상 0.
    """

    def __init__(self, prototypes: dict, modality: list,
                 device: str = "cuda", gamma: float = 4.0, alpha_temp: float = 3.0,
                 raw_logit_arrays: dict = None, dist_stats: dict = None, **kwargs):
        super().__init__(
            prototypes=prototypes,
            modality=modality,
            device=device,
            beta=0.0,   # always 0: no distance penalty
            gamma=gamma,
            alpha_temp=alpha_temp,
            raw_logit_arrays=raw_logit_arrays,
            dist_stats=dist_stats,
        )
