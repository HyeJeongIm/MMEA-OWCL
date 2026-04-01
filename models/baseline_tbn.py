# models/baseline.py

import torch
import torch.nn as nn
import copy
import logging

from models.backbones import get_backbone
from models.fusion import get_fusion
from models.classifier.classification_tbn import TBNClassification


# =============================================================================
# [Logit-Diag] 첫 eval 배치의 logit → energy → delta → r_m → α 전체 흐름 출력
# =============================================================================

_MMEA_CLASS_NAMES = {
    0: "upstairs",       1: "downstairs",     2: "drinking",       3: "fall",
    4: "reading",        5: "sweep_floor",     6: "cut_fruits",     7: "mop_floor",
    8: "writing",        9: "wipe_table",     10: "wash_hand",     11: "standing",
   12: "play_phone",    13: "type_pc",        14: "eating",        15: "cooking",
   16: "pick_up_phone", 17: "drop_trush",     18: "fold_clothes",  19: "walking",
   20: "play_card",     21: "brush_teeth",    22: "wash_dish",     23: "moving_sth",
   24: "type_phone",    25: "chat",           26: "open_close_door",27: "ride_bike",
   28: "sit_stand",     29: "take_drop_sth",  30: "shopping",      31: "watch_TV",
}


def _cls_name(idx):
    """class index → 이름 (없으면 index 그대로)"""
    return _MMEA_CLASS_NAMES.get(int(idx), str(idx))


def _diag_batch(main_logits, aux_logits, confidences, targets,
                energy_norm_method='?', max_samples=8, batch_idx=0, phase='ID',
                log_file=None):
    """
    하나의 배치에서 샘플별로 전체 계산 흐름을 출력합니다.

    출력 순서 (샘플마다):
      ① aux logit 벡터 (각 modality)
      ② main logit 벡터 (주 분류기)
      ③ energy E_m = -logsumexp(aux_logit),  E_main
      ④ E_mean (aux), delta = (E_m - E_mean) / T
      ⑤ r_m (forward 실제값) → softmax α → dominant modality
      ⑥ main pred vs aux pred 비교

    Args:
        log_file: 파일 경로 (str|None). 지정하면 stdout 출력과 동시에 파일에 append.
    """
    import os

    mods = list(aux_logits.keys())
    B    = main_logits.shape[0]
    n    = min(B, max_samples)
    T    = 1.0

    phase_tag = "◆ OOD" if phase == 'OOD' else "◇ ID "
    BAR = ("█" if phase == 'OOD' else "═") * 120

    # ── 출력 헬퍼: stdout + 파일 동시 기록 ──────────────────────────────────
    _buf = []
    def emit(s=""):
        print(s)
        _buf.append(s + "\n")

    def flush_to_file():
        if log_file and _buf:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.writelines(_buf)
    # ─────────────────────────────────────────────────────────────────────────

    emit(f"\n{BAR}")
    emit(f"  [Logit-Diag] {phase_tag}  Eval batch {batch_idx}  │  B={B} (showing {n})  │"
         f"  energy_norm_method={energy_norm_method}  │  T={T}")
    emit(BAR)

    for i in range(n):
        lbl_idx = targets[i].item() if targets is not None else None
        lbl_str = f"{lbl_idx} ({_cls_name(lbl_idx)})" if lbl_idx is not None else "?"
        emit(f"\n  ┌─ Sample {i:>2}  label={lbl_str} " + "─" * 70)

        # ① aux logit 벡터
        for m in mods:
            vec      = aux_logits[m][i].float().tolist()
            pred_idx = int(torch.tensor(vec).argmax())
            vals     = "  ".join(f"{v:+6.3f}" for v in vec)
            correct  = "✓" if pred_idx == lbl_idx else "✗"
            emit(f"  │  aux logit [{m:>5}] = [{vals}]   argmax={pred_idx} ({_cls_name(pred_idx)}) {correct}")

        # ② main logit 벡터
        main_vec      = main_logits[i].float().tolist()
        main_pred_idx = int(torch.tensor(main_vec).argmax())
        main_vals     = "  ".join(f"{v:+6.3f}" for v in main_vec)
        main_correct  = "✓" if main_pred_idx == lbl_idx else "✗"
        emit(f"  │  main logit         = [{main_vals}]   argmax={main_pred_idx} ({_cls_name(main_pred_idx)}) {main_correct}")
        emit(f"  │")

        # ③ energy (aux + main)
        e      = {m: -torch.logsumexp(aux_logits[m][i].float(), dim=0).item() for m in mods}
        e_main = -torch.logsumexp(main_logits[i].float(), dim=0).item()
        e_mean = sum(e[m] for m in mods) / len(mods)
        e_str  = "   ".join(f"E_{m}={e[m]:+.4f}" for m in mods)
        emit(f"  │  {e_str}   →  E_mean={e_mean:+.4f}   E_main={e_main:+.4f}")
        emit(f"  │  ↑ E = -logsumexp(logit)   낮을수록 확실")

        # ④ delta (aux 간 상대값)
        delta = {m: (e[m] - e_mean) / T for m in mods}
        d_str = "   ".join(f"Δ_{m}={delta[m]:+.4f}" for m in mods)
        emit(f"  │  {d_str}")
        emit(f"  │  ↑ Δ = (E_m − E_mean)/T   음수=확실  양수=불확실")
        emit(f"  │")

        # ⑤⑥ r_m → α 비교표
        emit(f"  │  {'method':<16}  " +
             "  ".join(f"{'r_'+m:<12}" for m in mods) +
             "  │  " + "  ".join(f"{'α_'+m:<10}" for m in mods) +
             "  dominant   correct?")
        emit(f"  │  {'─'*16}  " + "  ".join("─" * 12 for _ in mods) +
             "  │  " + "  ".join("─" * 10 for _ in mods))

        # 실제 r_m (forward에서 계산된 값)
        if confidences:
            r_real = [confidences[m][i].item() for m in mods]
            r_t    = torch.tensor(r_real)
            alpha  = torch.softmax(r_t, dim=0)
            dom    = mods[alpha.argmax().item()]
            ok     = "✓" if dom == mods[torch.tensor([e[m] for m in mods]).argmin().item()] else "✗"
            r_str  = "  ".join(f"{r_real[j]:+.4f}      " for j in range(len(mods)))
            a_str  = "  ".join(f"{alpha[j]:.4f}    " for j in range(len(mods)))
            emit(f"  │  {'[actual r_m]':<16}  {r_str}  │  {a_str}  {dom:<10} {ok}")

        # high_sigmoid 재현
        r_hs  = torch.tensor([torch.sigmoid(torch.tensor(delta[m])).item() for m in mods])
        alp_hs = torch.softmax(r_hs, dim=0)
        dom_hs = mods[alp_hs.argmax().item()]
        r_str_hs = "  ".join(f"{r_hs[j]:+.4f}      " for j in range(len(mods)))
        a_str_hs = "  ".join(f"{alp_hs[j]:.4f}    " for j in range(len(mods)))
        tag_hs   = " ← current" if energy_norm_method == "high_sigmoid" else ""
        emit(f"  │  {'high_sigmoid':<16}  {r_str_hs}  │  {a_str_hs}  {dom_hs:<10}{tag_hs}")

        # ⑦ 예측 일치 요약
        aux_preds = {m: int(aux_logits[m][i].float().argmax()) for m in mods}
        agree     = all(v == main_pred_idx for v in aux_preds.values())
        pred_str  = "  ".join(f"{m}={aux_preds[m]}({_cls_name(aux_preds[m])})" for m in mods)
        emit(f"  │")
        emit(f"  │  pred: main={main_pred_idx}({_cls_name(main_pred_idx)})  "
             f"aux=[{pred_str}]")
        emit(f"  │  label={lbl_str}  all_agree={'✓' if agree else '✗'}")
        emit(f"  └{'─'*118}")

    emit(f"\n{BAR}\n")
    flush_to_file()


# =============================================================================


class TBNBaseline(nn.Module):
    """Multi-modal baseline network with backbone, fusion, and classifier"""
    
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_segments = args["num_segments"]
        self.modality = args["modality"]
        self.backbone_name = args["backbone"]  # e.g., 'tbn'
        self.fusion_type = args.get("fusion_type", args.get("midfusion", "concat"))  # 'concat', 'attention', etc.
        self.dropout = args["dropout"]
        self.consensus_type = args["consensus_type"]
        self.before_softmax = args["before_softmax"]

        if not self.before_softmax and self.consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        # Initialize backbone network for feature extraction
        self.backbone = get_backbone(args)  # output: feature list per modality

        # Initialize fusion network to combine multi-modal features
        self.fusion = get_fusion(
            midfusion=self.fusion_type,
            feature_dim=self.backbone.feature_dim, # 각 모달리티마다 1024
            modality=self.modality,
            dropout=self.dropout,
            num_segments=self.num_segments,
            shared_dim=args.get("shared_dim", 256),  # JSON에서 설정 가능
            num_classes=args.get("init_cls", 8),  # 초기 클래스 수
            consensus_type=self.consensus_type,  # TBN consensus 방법
            before_softmax=self.before_softmax,   # TBN softmax 옵션
            pretrain_epochs=args.get("pretrain_epochs", None),  # Auxiliary head pretrain epochs (JSON에서 설정 가능)
            confidence_method=args.get("confidence_method", "max_prob"),  # Confidence 계산 방법 (JSON에서 설정 가능)
            aux_loss_weight=args.get("morst_lambda", args.get("aux_loss_weight", 0.5)),  # λ (morst_lambda for MAND, aux_loss_weight for others)
            energy_norm_method=args.get("energy_norm_method", "zscore"),  # Energy 정규화 방법 (JSON에서 설정 가능)
        )

        # Set final feature dimension based on modality count
        # Follows the same logic as original Baseline class
        if len(self.modality) > 1:
            self.feature_dim = 512  # Multi-modal fusion output
        else:
            self.feature_dim = self.backbone.feature_dim  # Single modality: keep original dimension
            
        # Debug: Print feature dimension info
        print(f"🔍 BaselineTBN Debug:")
        print(f"   Modality count: {len(self.modality)}")
        print(f"   Backbone feature_dim: {self.backbone.feature_dim}")
        print(f"   After fusion feature_dim: {self.feature_dim}")
        self.fc = None  # Classifier will be created via update_fc()
        self._logit_diag_batches_shown = 0   # ID: 현재 task에서 출력한 배치 수
        self._logit_diag_ood_shown     = 0   # OOD: 현재 task에서 출력한 배치 수
        self._logit_diag_max_batches   = 3   # task당 최대 출력 배치 수
        self._diag_phase = 'ID'              # 현재 diagnostic phase ('ID' or 'OOD')

        # Pass num_segments to fusion if it supports TBN (auxiliary_head, auxiliary_head_v2 등)
        if hasattr(self.fusion, 'num_segments'):
            self.fusion.num_segments = self.num_segments
            print(f"🔧 Set fusion.num_segments = {self.num_segments}")

        print("=" * 40)
        print("✅ Baseline Model Configuration")
        print("-" * 40)
        print(f"  Backbone:        {self.backbone_name}")
        print(f"  Fusion:          {self.fusion_type}")
        print(f"  Modality:        {self.modality}")
        print(f"  Segments:        {self.num_segments}")
        print(f"  Dropout:         {self.dropout}")
        print(f"  Consensus:       {self.consensus_type}")
        print("=" * 40)

    @property
    def output_dim(self):
        """Return output feature dimension"""
        return self.feature_dim

    def extract_vector(self, x):
        """Forward pass: backbone -> fusion"""
        features = self.backbone(x)  # Extract per-modality features
        fused = self.fusion(features)  # Fuse multi-modal features
        return fused["features"]

    def forward(self, x, targets=None):
        """Forward pass: backbone -> fusion -> classifier"""
        features = self.backbone(x)  # Extract features from each modality
        fused = self.fusion(features, targets=targets)  # Combine features across modalities (with targets for auxiliary loss)
        out = self.fc(fused["features"])  # Apply classifier
        out.update(fused)  # Include fusion output

        # 🎯 Auxiliary loss를 최상위로 이동 (학습 루프에서 쉽게 접근 가능)
        if 'auxiliary_loss' in fused:
            out['auxiliary_logits'] = fused['auxiliary_logits']
            out['auxiliary_loss'] = fused['auxiliary_loss']
            out['aux_loss_weight'] = fused.get('aux_loss_weight', 0.0)

        # 🔬 [Logit-Diag] eval 배치별 전체 흐름 출력 (debug_mode 시, task당 max_batches회)
        _phase    = self._diag_phase
        _shown    = self._logit_diag_ood_shown if _phase == 'OOD' else self._logit_diag_batches_shown
        if (not self.training
                and _shown < self._logit_diag_max_batches
                and self.args.get('debug_mode', False)
                and 'auxiliary_logits' in out
                and out['auxiliary_logits']):
            _diag_batch(
                main_logits=out['logits'].detach().cpu(),
                aux_logits={m: v.detach().cpu() for m, v in out['auxiliary_logits'].items()},
                confidences={m: v.detach().cpu() for m, v in out.get('confidences', {}).items()},
                targets=targets.cpu() if targets is not None else None,
                energy_norm_method=self.args.get('energy_norm_method', '?'),
                batch_idx=_shown,
                phase=_phase,
                log_file=self.args.get('diag_log_path', None),
            )
            if _phase == 'OOD':
                self._logit_diag_ood_shown += 1
            else:
                self._logit_diag_batches_shown += 1

        return out

    def update_fc(self, nb_classes):
        """Update classifier for new number of classes while preserving weights"""
        # Create new classifier with updated class count
        new_fc = TBNClassification(
            feature_dim=self.feature_dim,
            modality=self.modality,
            num_class=nb_classes,
            consensus_type=self.consensus_type,
            before_softmax=self.before_softmax,
            num_segments=self.num_segments
        )

        # Preserve existing classifier weights if available
        if self.fc is not None:
            nb_output = self.fc.num_class
            new_fc.fc_action.weight.data[:nb_output] = self.fc.fc_action.weight.data
            new_fc.fc_action.bias.data[:nb_output] = self.fc.fc_action.bias.data

        self.fc = new_fc
        self._logit_diag_batches_shown = 0  # task 바뀌면 다시 출력
        self._logit_diag_ood_shown     = 0
        self._diag_phase = 'ID'

        # Update fusion auxiliary heads if available
        if hasattr(self.fusion, 'update_auxiliary_heads'):
            self.fusion.update_auxiliary_heads(nb_classes)
            logging.info(f"🎯 Updated fusion auxiliary heads to {nb_classes} classes")

    def copy(self):
        """Create deep copy of the model"""
        return copy.deepcopy(self)

    def freeze(self):
        """Freeze all parameters for inference"""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self
