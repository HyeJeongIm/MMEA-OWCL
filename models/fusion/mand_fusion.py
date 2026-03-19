"""
MANDFusion: MAND Fusion Module
================================
Single fusion module that implements both components of MAND:

  MoRST (training-time):
      Equips each modality encoder with a lightweight head H_m(·) and
      computes per-modality supervision + logit-distillation loss.

      L_Sup = L_CE(z_main, y) + λ · (1/|M|) Σ_m L_CE(z_m, y)

  MoAS (inference-time):
      Estimates sample-wise modality reliability from energy scores and
      adaptively integrates weighted modality logits into the main logits.

      E_m(x) = -log Σ_c exp(z_{m,c})          (Eq. 1)
      r_m(x) = -(E_m(x) - μ_m) / σ_m          (Eq. 2)
      α_m(x) = exp(r_m) / Σ_j exp(r_j)        (Eq. 3)
      z^MoAS = z_main + Σ_m α_m · z_m         (Eq. 4)
      s(x)   = max_c z^MoAS                    (Eq. 5)

Interface compatibility with mmeabase.py / baseline_tbn.py:
  forward() output dict keeps:
      'auxiliary_logits'   {m: [B,C]} z_m       ← mmeabase, OOD detector
      'auxiliary_loss'     Tensor|None           ← mmeabase._compute_total_loss
      'aux_loss_weight'    float  λ              ← mmeabase, baseline_tbn
      'is_pretrain_phase'  bool                  ← mmeabase
  attribute  auxiliary_heads  (property alias for modality_heads)
  method     update_auxiliary_heads(nb_classes)  ← baseline_tbn.update_fc
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, normal_

from utils.basic_ops import ConsensusModule


class MANDFusion(nn.Module):
    """
    MAND Fusion Module — MoRST (training) + MoAS (inference).

    Pretrain phase (epoch 0 .. pretrain_epochs-1 in every task):
        - Modality heads H_m are trained with cross-entropy gradient.
        - auxiliary_loss = (1/|M|) Σ_m L_CE(z_m, y)  is returned.

    Post-pretrain phase (epoch ≥ pretrain_epochs):
        - Modality heads are frozen; z_m and r_m computed under no_grad.
        - moas_score() can be called to produce the full MoAS novelty score.
    """

    def __init__(
        self,
        feature_dim: int,
        modality: list,
        dropout: float,
        num_classes: int = 32,
        confidence_method: str = "energy",
        aux_loss_weight: float = 0.5,
        consensus_type: str = "avg",
        before_softmax: bool = True,
        num_segments: int = 8,
        pretrain_epochs: int = 5,
        energy_norm_method: str = "zscore",
    ):
        """
        Args:
            feature_dim:        per-modality feature dimension (e.g. 1024 for TBN)
            modality:           ordered list, e.g. ['RGB', 'Gyro', 'Acce']
            dropout:            dropout probability for the fusion MLP
            num_classes:        initial output dimension of modality heads H_m(·)
            confidence_method:  confidence method for _compute_confidence
                                ('energy' + 'zscore' → MoAS r_m; others for ablation)
            aux_loss_weight:    λ — weight on per-modality head supervision loss
            consensus_type:     TBN consensus ('avg' or 'identity')
            before_softmax:     whether logits are pre-softmax
            num_segments:       number of TBN segments
            pretrain_epochs:    epochs per task to train modality heads
            energy_norm_method: energy normalisation ('zscore', 'e_uniform', 'raw')
        """
        super().__init__()

        self.modality = modality
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.confidence_method = confidence_method
        self.morst_lambda = aux_loss_weight  # λ — paper name internally
        self.energy_norm_method = energy_norm_method

        self.consensus_type = consensus_type
        self.before_softmax = before_softmax
        self.num_segments = num_segments
        self.reshape = True

        self.modality_to_idx = {m: i for i, m in enumerate(self.modality)}

        # Pretrain / freeze bookkeeping
        self.pretrain_epochs = pretrain_epochs
        self.current_epoch = 0
        self.current_task_id = 0
        self.auxiliary_heads_frozen = False

        if len(self.modality) <= 1:
            raise ValueError("MANDFusion requires at least two modalities")

        # ── Modality-specific heads  H_m(·) ──────────────────────────────
        # Internal name: modality_heads
        # External name: auxiliary_heads  (property alias — required by mmeabase.py)
        self.modality_heads = nn.ModuleDict()
        for m in self.modality:
            self.modality_heads[m] = nn.Linear(feature_dim, num_classes)
            normal_(self.modality_heads[m].weight, 0, 0.001)
            constant_(self.modality_heads[m].bias, 0)

        # ── Fusion MLP: uniform concat → fc1(512) → ReLU → dropout ───────
        input_dim = len(self.modality) * feature_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        normal_(self.fc1.weight, 0, 0.001)
        constant_(self.fc1.bias, 0)
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.first_forward_per_task: dict = {}
        self.epoch_logged: set = set()

        # Per-modality energy statistics {m: (μ, σ)} — injected by mmeabase
        # via _compute_energy_stats_from_memory → set_energy_stats()
        self._energy_stats: dict = {}

        self.consensus = ConsensusModule(consensus_type)
        if not self.before_softmax:
            self.softmax = nn.Softmax(dim=1)

    # ------------------------------------------------------------------
    # mmeabase / baseline_tbn compatibility interface  (checklist E3, E4)
    # ------------------------------------------------------------------

    @property
    def auxiliary_heads(self) -> nn.ModuleDict:
        """
        Alias for modality_heads.
        Required by mmeabase.py:
            hasattr(fusion, 'auxiliary_heads')   → triggers energy stats computation
            hasattr(fusion_module, 'auxiliary_heads')  → confidence collection
        """
        return self.modality_heads

    def update_auxiliary_heads(self, nb_classes: int):
        """
        Expand modality head output dimension for new task classes.
        Called by baseline_tbn.TBNBaseline.update_fc() (baseline_tbn.py:128).
        Old weights are preserved; new class rows are near-zero initialised.
        """
        old_num_classes = self.num_classes
        self.num_classes = nb_classes
        for m in self.modality:
            old_head = self.modality_heads[m]
            new_head = nn.Linear(self.feature_dim, nb_classes)
            if old_num_classes > 0:
                new_head.weight.data[:old_num_classes] = old_head.weight.data
                new_head.bias.data[:old_num_classes] = old_head.bias.data
            if nb_classes > old_num_classes:
                normal_(new_head.weight.data[old_num_classes:], 0, 0.001)
                constant_(new_head.bias.data[old_num_classes:], 0)
            self.modality_heads[m] = new_head
        logging.info(
            f"[MANDFusion] Modality heads updated: {old_num_classes} → {nb_classes} classes"
        )

    # ------------------------------------------------------------------
    # Energy statistics injection  (called by mmeabase._compute_energy_stats_from_memory)
    # ------------------------------------------------------------------

    def set_energy_stats(self, energy_stats: dict):
        """
        Inject per-modality energy statistics (μ_m, σ_m) from the replay buffer.
        Required for MoAS z-score reliability computation (Eq. 2).

        Args:
            energy_stats: {modality_name: (mean: float, std: float)}
        """
        self._energy_stats = energy_stats
        logging.info("[MANDFusion] MoAS energy statistics injected:")
        for mod, (mean, std) in energy_stats.items():
            logging.info(f"  {mod}: μ={mean:.4f}, σ={std:.4f}")

    # ------------------------------------------------------------------
    # Epoch / task management
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        if epoch == self.pretrain_epochs and not self.auxiliary_heads_frozen:
            self._freeze_modality_heads()
            if epoch not in self.epoch_logged:
                logging.info(
                    f"[MANDFusion] Task {self.current_task_id}: "
                    f"modality heads frozen at epoch {epoch} (MoRST pretrain complete)"
                )
                self.epoch_logged.add(epoch)

    def update_task(self, task_id: int):
        """Reset pretrain state and unfreeze heads for the new task."""
        self.current_task_id = task_id
        self.current_epoch = 0
        self.auxiliary_heads_frozen = False
        self.epoch_logged.clear()
        for mod in self.modality:
            setattr(self, f"_energy_logged_{mod}", False)
        for m in self.modality:
            for p in self.modality_heads[m].parameters():
                p.requires_grad = True
        if task_id not in self.first_forward_per_task:
            self.first_forward_per_task[task_id] = True
        logging.info(
            f"[MANDFusion] Task {task_id}: modality heads unfrozen "
            f"for MoRST pretrain (epochs 0–{self.pretrain_epochs - 1})"
        )

    def _freeze_modality_heads(self):
        for m in self.modality:
            for p in self.modality_heads[m].parameters():
                p.requires_grad = False
        self.auxiliary_heads_frozen = True
        logging.info(f"[MANDFusion] Modality heads frozen: {list(self.modality_heads.keys())}")

    def _is_pretrain_phase(self) -> bool:
        return self.current_epoch < self.pretrain_epochs

    # ------------------------------------------------------------------
    # MoRST: confidence / reliability computation  (r_m, Eq. 2)
    # ------------------------------------------------------------------

    def _compute_confidence(self, logits: torch.Tensor, modality_name: str = None) -> torch.Tensor:
        """
        Compute per-sample confidence from modality head logits.

        When confidence_method == 'energy' and energy_norm_method == 'zscore',
        returns r_m = -(E_m - μ_m) / σ_m  (Eq. 2) — exactly what MoAS uses.
        """
        if self.confidence_method == "energy":
            energy = -torch.logsumexp(logits, dim=1)  # E_m (Eq. 1)

            if self.energy_norm_method == "zscore":
                stats = self._energy_stats.get(modality_name) if modality_name else None
                if stats is not None:
                    e_mean, e_std = stats
                    return -(energy - e_mean) / (e_std + 1e-8)  # r_m (Eq. 2)
                return torch.sigmoid(-energy)  # fallback: stats not yet injected

            elif self.energy_norm_method == "e_uniform":
                num_cls = logits.shape[1]
                e_uniform = -torch.log(
                    torch.tensor(num_cls, dtype=torch.float32, device=logits.device)
                )
                return torch.sigmoid(-(energy - e_uniform))

            else:  # "raw"
                return torch.sigmoid(-energy)

        elif self.confidence_method == "entropy":
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = torch.log(
                torch.tensor(self.num_classes, device=entropy.device, dtype=torch.float32)
            )
            return 1.0 - (entropy / max_entropy)

        elif self.confidence_method == "max_prob":
            probs = F.softmax(logits, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            return confidence

        else:
            return torch.ones(logits.size(0), device=logits.device) / len(self.modality)

    # ------------------------------------------------------------------
    # MoAS: full inference-time novelty scoring  (Eqs. 2-5)
    # ------------------------------------------------------------------

    def moas_score(self, z_main: torch.Tensor, modality_logits: dict) -> dict:
        """
        Compute MoAS combined logits and novelty score.

        z^MoAS_c = z_main,c + Σ_m α_m · z_m,c    (Eq. 4)
        s(x)     = max_c z^MoAS_c                 (Eq. 5)

        Args:
            z_main:          [B, C]  main classifier logits
            modality_logits: {m: Tensor [B, C]}  z_m from H_m

        Returns:
            dict:
                'combined_logits':    [B, C]   z^MoAS
                'novelty_score':      [B]      s(x)
                'weights':            {m: [B]} α_m per modality
                'reliability_scores': {m: [B]} r_m per modality
                'energy_scores':      {m: [B]} E_m per modality
        """
        energy_dict = {}
        reliability_dict = {}

        for m in self.modality:
            if m not in modality_logits or modality_logits[m] is None:
                continue
            z_m = modality_logits[m]
            energy = -torch.logsumexp(z_m, dim=1)              # E_m (Eq. 1)
            reliability = self._compute_confidence(z_m, m)     # r_m (Eq. 2)
            energy_dict[m] = energy
            reliability_dict[m] = reliability

        # No valid modalities: return main logits unchanged
        if not reliability_dict:
            return {
                "combined_logits": z_main,
                "novelty_score": z_main.max(dim=1).values,
                "weights": {},
                "reliability_scores": {},
                "energy_scores": {},
            }

        # α_m = softmax({r_m})  (Eq. 3)
        modalities = list(reliability_dict.keys())
        r_stack = torch.stack([reliability_dict[m] for m in modalities], dim=1)  # [B, M]
        alpha_stack = F.softmax(r_stack, dim=1)                                  # [B, M]
        weights = {m: alpha_stack[:, i] for i, m in enumerate(modalities)}

        # z^MoAS = z_main + Σ_m α_m · z_m  (Eq. 4)
        combined = z_main.clone()
        for m, alpha in weights.items():
            z_m = modality_logits[m]
            # Align class dimension (z_m may have fewer classes if from earlier task)
            if z_m.shape[1] < z_main.shape[1]:
                pad = torch.zeros(
                    z_m.shape[0], z_main.shape[1] - z_m.shape[1],
                    device=z_m.device, dtype=z_m.dtype,
                )
                z_m = torch.cat([z_m, pad], dim=1)
            elif z_m.shape[1] > z_main.shape[1]:
                z_m = z_m[:, :z_main.shape[1]]
            combined = combined + alpha.unsqueeze(1) * z_m

        # s(x) = max_c z^MoAS  (Eq. 5)
        novelty_score = combined.max(dim=1).values

        return {
            "combined_logits": combined,
            "novelty_score": novelty_score,
            "weights": weights,
            "reliability_scores": reliability_dict,
            "energy_scores": energy_dict,
        }

    # ------------------------------------------------------------------
    # TBN helpers
    # ------------------------------------------------------------------

    def _apply_consensus_to_logits(self, aux_logits: torch.Tensor) -> torch.Tensor:
        if self.num_segments <= 1:
            return aux_logits
        if aux_logits.size(0) % self.num_segments != 0:
            return aux_logits
        base_out = aux_logits
        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        output = self.consensus(base_out)
        if self.consensus_type == "identity":
            return output[:, 0, :]
        return output.squeeze(1)

    def _pick_features(self, features):
        f_rgb  = features[self.modality_to_idx["RGB"]]  if "RGB"  in self.modality_to_idx else None
        f_gyro = features[self.modality_to_idx["Gyro"]] if "Gyro" in self.modality_to_idx else None
        f_acce = features[self.modality_to_idx["Acce"]] if "Acce" in self.modality_to_idx else None
        return f_rgb, f_gyro, f_acce

    # ------------------------------------------------------------------
    # Forward  (MoRST training pass)
    # ------------------------------------------------------------------

    def forward(self, features, targets=None):
        """
        Forward pass: MoRST feature extraction + head supervision.

        Pretrain phase (epoch 0 .. pretrain_epochs-1):
            - Modality heads computed with gradient.
            - auxiliary_loss = (1/|M|) Σ_m L_CE(z_m, y)

        Post-pretrain phase:
            - Modality heads computed under no_grad (frozen inference).
            - confidences = r_m values for MoAS at OOD evaluation time.

        Returns dict (keys preserved for mmeabase / OOD detector compatibility):
            'features'              [B, 512]   fused representation
            'auxiliary_logits'      {m: [B,C]} z_m — for MoAS + mmeabase
            'auxiliary_loss'        Tensor|None — mmeabase._compute_total_loss
            'aux_loss_weight'       float  λ (morst_lambda) — mmeabase key
            'confidences'           {m: [B]}   r_m reliability — OOD detector
            'is_pretrain_phase'     bool
            'auxiliary_heads_frozen' bool
            'fusion_type'           str
        """
        f_rgb, f_gyro, f_acce = self._pick_features(features)
        raw = {"RGB": f_rgb, "Gyro": f_gyro, "Acce": f_acce}
        modality_features = {m: raw[m] for m in self.modality if raw.get(m) is not None}

        is_pretrain = self._is_pretrain_phase()
        auxiliary_logits: dict = {}
        confidences: dict = {}

        # ── Modality heads z_m + reliability r_m ─────────────────────────
        if is_pretrain:
            for m, feature in modality_features.items():
                z_m_seg = self.modality_heads[m](feature)
                z_m = self._apply_consensus_to_logits(z_m_seg)
                auxiliary_logits[m] = z_m
                confidences[m] = self._compute_confidence(z_m, modality_name=m)
        else:
            with torch.no_grad():
                for m, feature in modality_features.items():
                    z_m_seg = self.modality_heads[m](feature)
                    z_m = self._apply_consensus_to_logits(z_m_seg)
                    auxiliary_logits[m] = z_m
                    confidences[m] = self._compute_confidence(z_m, modality_name=m)

        # ── Uniform 1:1:1 feature fusion → fc1 → ReLU → dropout ──────────
        weighted_features = [modality_features[m] for m in self.modality if m in modality_features]
        x = torch.cat(weighted_features, dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)

        # ── MoRST head supervision loss  (1/|M|) Σ_m L_CE(z_m, y) ───────
        auxiliary_loss = None
        if is_pretrain and targets is not None and auxiliary_logits:
            auxiliary_loss = sum(
                F.cross_entropy(z_m, targets) for z_m in auxiliary_logits.values()
            ) / len(auxiliary_logits)

        if self.first_forward_per_task.get(self.current_task_id, False):
            phase_str = "MoRST-Pretrain" if is_pretrain else "MoRST-Frozen / MoAS-Ready"
            logging.info(
                f"[MANDFusion] Task {self.current_task_id}, Epoch {self.current_epoch}: "
                f"phase={phase_str}, modalities={list(auxiliary_logits.keys())}, λ={self.morst_lambda}"
            )
            self.first_forward_per_task[self.current_task_id] = False

        return {
            "features": x,
            "auxiliary_logits": auxiliary_logits,    # z_m — kept for mmeabase / MoAS
            "auxiliary_loss": auxiliary_loss,         # kept for mmeabase
            "aux_loss_weight": self.morst_lambda,     # λ — kept as 'aux_loss_weight' for mmeabase
            "confidences": confidences,               # r_m — used by OOD detector for MoAS
            "is_pretrain_phase": is_pretrain,
            "auxiliary_heads_frozen": self.auxiliary_heads_frozen,
            "fusion_type": "mand_fusion",
        }

    def compute_total_loss(self, main_loss, auxiliary_loss=None):
        """
        L_Sup = L_CE(z_main, y) + λ · (1/|M|) Σ_m L_CE(z_m, y)
        Called by mmeabase._compute_total_loss during the pretrain phase.
        """
        if not self._is_pretrain_phase() or auxiliary_loss is None or self.morst_lambda == 0:
            return main_loss
        return main_loss + self.morst_lambda * auxiliary_loss
