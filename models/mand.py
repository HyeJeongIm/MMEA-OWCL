"""
MAND: Modality-Aware Novelty Detection and Training
=====================================================
Full continual learning model implementing the paper's method.

  MoRST (training):  preserves per-modality decision boundaries across tasks
                     via modality-specific heads + logit distillation on replayed exemplars.

  MoAS  (inference): adaptively integrates weighted modality logits into the
                     main logits for energy-based novelty scoring.

Loss (Task t > 0):
    L = L_Sup + β · L_KD

    L_Sup = L_CE(z_main, y) + λ · (1/|M|) Σ_m L_CE(z_m, y)    (head supervision)
    L_KD  = Σ_m || z̃_m - z_m ||²                                (modality-wise KD)

where:
    β  = morst_beta   (distillation weight, paper name)
    λ  = morst_lambda (head supervision weight, stored in args)
    z̃_m = stored modality logits from the replay buffer (_modality_logits_memory)
"""

import copy
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline_tbn import TBNBaseline
from models.baseline_tsn import TSNBaseline
from models.replay import Replay
from utils.toolkit import tensor2numpy

EPSILON = 1e-8


class MAND(Replay):
    """
    MAND: Modality-Aware Novelty Detection and Training.

    Inherits from Replay (exemplar management, basic CL loop).
    Overrides training loops to add MoRST modality-wise KD loss.
    """

    def __init__(self, args):
        super().__init__(args)

        # β — distillation weight for L_KD  (paper name: morst_beta)
        self.morst_beta = args.get("morst_beta", 0.08)

        # MoRST 활성화 여부 (wo_morst ablation용)
        self.morst_enabled = args.get("morst_enabled", True)

        # z̃_m buffer: stored modality logits per exemplar
        # {modality: np.ndarray [N, C]}
        self._modality_logits_memory = defaultdict(lambda: np.array([]))

        logging.info(f"[MAND] Initialized: β={self.morst_beta}, morst_enabled={self.morst_enabled}")

    # ------------------------------------------------------------------
    # Incremental training entry point
    # ------------------------------------------------------------------

    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._data_manager = data_manager

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._classes_seen_so_far = self._total_classes
        self.class_increments.append([self._known_classes, self._total_classes - 1])

        self._network.update_fc(self._total_classes)
        logging.info(f"[MAND] Learning on {self._known_classes}-{self._total_classes}")

        self._setup_data_loaders_with_ood(data_manager)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # Store old modality logits for replay (MoRST KD)
        if self._cur_task > 0 and hasattr(self, "_data_memory") and self._data_memory.size > 0:
            if self.morst_enabled:
                logging.info("[MAND] Setting up MoRST train loaders with stored modality logits...")
                self._setup_morst_train_loaders(data_manager)
            else:
                logging.info("[MAND w/o MoRST] Skipping MoRST KD — using basic replay (no modality logits)")

        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._build_knn_dist_ref_from_memory(data_manager)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    # ------------------------------------------------------------------
    # DataLoader setup for MoRST replay
    # ------------------------------------------------------------------

    def _setup_morst_train_loaders(self, data_manager):
        """
        Build the training DataLoader that includes stored modality logits z̃_m
        so _update_representation can compute L_KD = Σ_m || z̃_m - z_m ||².
        """
        logging.info(f"[MAND] Setting up MoRST train loaders (Task {self._cur_task})")

        memory_data = self._get_memory()
        if (
            memory_data is not None
            and self._cur_task > 0
            and hasattr(self, "_modality_logits_memory")
        ):
            if isinstance(self._modality_logits_memory, dict):
                appendent = (memory_data[0], memory_data[1], self._modality_logits_memory)
            elif len(self._modality_logits_memory) > 0:
                appendent = (memory_data[0], memory_data[1], self._modality_logits_memory)
            else:
                appendent = memory_data
        else:
            appendent = memory_data

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=appendent,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    # ------------------------------------------------------------------
    # Exemplar management (override from Replay — adds modality logit handling)
    # ------------------------------------------------------------------

    def _reduce_exemplar_reservoir(self, data_manager, m):
        """Reduce exemplars with Reservoir Sampling; keep modality logits aligned."""
        logging.info(f"[MAND] Reducing exemplars with Reservoir Sampling ({m} per class)")

        dummy_data = copy.deepcopy(self._data_memory)
        dummy_targets = copy.deepcopy(self._targets_memory)
        dummy_logits = copy.deepcopy(self._modality_logits_memory)

        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._modality_logits_memory = defaultdict(lambda: np.array([]))

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            class_data = dummy_data[mask]

            m_current = min(m, len(class_data))
            if m_current == 0:
                continue

            indices = np.random.choice(len(class_data), size=m_current, replace=False)
            dd = class_data[indices]
            dt = dummy_targets[mask][indices]

            self._data_memory = (
                np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            for mod in self._modality:
                if (
                    isinstance(dummy_logits, dict)
                    and mod in dummy_logits
                    and len(dummy_logits[mod]) > 0
                ):
                    dl = dummy_logits[mod][mask][indices]
                    self._modality_logits_memory[mod] = (
                        np.concatenate((self._modality_logits_memory[mod], dl), axis=0)
                        if mod in self._modality_logits_memory
                        and len(self._modality_logits_memory[mod]) > 0
                        else dl
                    )

            logging.info(f"  Class {class_idx}: {m_current} exemplars kept")

    def _construct_exemplar_reservoir(self, data_manager, m):
        """
        Reservoir-sampled exemplar construction for new classes.
        Stores modality logits z̃_m alongside each selected exemplar.
        """
        logging.info(f"[MAND] Constructing exemplars with Reservoir Sampling ({m} per class)")

        # Pad existing logits to next task's class dimension
        if self._known_classes > 0:
            next_logits_dim = self._total_classes + self.args["increment"]
            for mod in self._modality:
                if mod in self._modality_logits_memory and len(self._modality_logits_memory[mod]) > 0:
                    cur_dim = self._modality_logits_memory[mod].shape[1]
                    if cur_dim < next_logits_dim:
                        self._modality_logits_memory[mod] = np.pad(
                            self._modality_logits_memory[mod],
                            ((0, 0), (0, next_logits_dim - cur_dim)),
                            mode="constant",
                            constant_values=0,
                        )

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
            )
            _, _, modality_logits_dict = self._extract_vectors_and_modality_logits(idx_loader)

            m_cur = min(m, data.shape[0])
            indices = np.random.choice(data.shape[0], size=m_cur, replace=False)
            selected_exemplars = data[indices]
            exemplar_targets = np.full(m_cur, class_idx)

            next_logits_dim = self._total_classes + self.args["increment"]
            for mod in self._modality:
                if mod in modality_logits_dict and len(modality_logits_dict[mod]) > 0:
                    exemplar_logits = modality_logits_dict[mod][indices]
                    cur_dim = exemplar_logits.shape[1]
                    if cur_dim < next_logits_dim:
                        exemplar_logits = np.pad(
                            exemplar_logits,
                            ((0, 0), (0, next_logits_dim - cur_dim)),
                            mode="constant",
                            constant_values=0,
                        )
                    self._modality_logits_memory[mod] = (
                        np.concatenate((self._modality_logits_memory[mod], exemplar_logits), axis=0)
                        if mod in self._modality_logits_memory
                        and len(self._modality_logits_memory[mod]) > 0
                        else exemplar_logits
                    )

            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
            logging.info(f"  Class {class_idx}: {m_cur} exemplars selected (Reservoir)")

    def _reduce_exemplar(self, data_manager, m):
        """Herding-based exemplar reduction; keeps modality logits aligned."""
        logging.info(f"[MAND] Reducing exemplars ({m} per class)")

        dummy_data = copy.deepcopy(self._data_memory)
        dummy_targets = copy.deepcopy(self._targets_memory)
        dummy_logits = copy.deepcopy(self._modality_logits_memory)

        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._modality_logits_memory = defaultdict(lambda: np.array([]))

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]

            self._data_memory = (
                np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            for mod in self._modality:
                if (
                    isinstance(dummy_logits, dict)
                    and mod in dummy_logits
                    and len(dummy_logits[mod]) > 0
                ):
                    dl = dummy_logits[mod][mask][:m]
                    self._modality_logits_memory[mod] = (
                        np.concatenate((self._modality_logits_memory[mod], dl), axis=0)
                        if mod in self._modality_logits_memory
                        and len(self._modality_logits_memory[mod]) > 0
                        else dl
                    )

            # Update class mean for NME
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        """
        Herding-based exemplar construction for new classes.
        Stores modality logits z̃_m alongside each selected exemplar.
        """
        logging.info(f"[MAND] Constructing exemplars ({m} per class)")

        # Pad existing logits to next task's class dimension
        if self._known_classes > 0:
            next_logits_dim = self._total_classes + self.args["increment"]
            for mod in self._modality:
                if mod in self._modality_logits_memory and len(self._modality_logits_memory[mod]) > 0:
                    cur_dim = self._modality_logits_memory[mod].shape[1]
                    if cur_dim < next_logits_dim:
                        self._modality_logits_memory[mod] = np.pad(
                            self._modality_logits_memory[mod],
                            ((0, 0), (0, next_logits_dim - cur_dim)),
                            mode="constant",
                            constant_values=0,
                        )

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
            )
            vectors, _, modality_logits_dict = self._extract_vectors_and_modality_logits(
                idx_loader
            )
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Herding selection
            selected_exemplars = []
            exemplar_vectors = []
            exemplar_logits_dict = defaultdict(list)

            m_cur = min(m, vectors.shape[0])
            for k in range(1, m_cur + 1):
                S = np.sum(exemplar_vectors, axis=0)
                mu_p = (vectors + S) / k
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(data[i])
                exemplar_vectors.append(vectors[i])

                for mod in self._modality:
                    if mod in modality_logits_dict:
                        exemplar_logits_dict[mod].append(modality_logits_dict[mod][i])

                vectors = np.delete(vectors, i, axis=0)
                data = np.delete(data, i, axis=0)
                for mod in self._modality:
                    if mod in modality_logits_dict:
                        modality_logits_dict[mod] = np.delete(
                            modality_logits_dict[mod], i, axis=0
                        )

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m_cur, class_idx)

            next_logits_dim = self._total_classes + self.args["increment"]
            for mod in self._modality:
                if mod in exemplar_logits_dict:
                    el = np.array(exemplar_logits_dict[mod])
                    cur_dim = el.shape[1]
                    if cur_dim < next_logits_dim:
                        el = np.pad(
                            el,
                            ((0, 0), (0, next_logits_dim - cur_dim)),
                            mode="constant",
                            constant_values=0,
                        )
                    self._modality_logits_memory[mod] = (
                        np.concatenate((self._modality_logits_memory[mod], el), axis=0)
                        if mod in self._modality_logits_memory
                        and len(self._modality_logits_memory[mod]) > 0
                        else el
                    )

            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Update class mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            self._class_means[class_idx, :] = mean

    # ------------------------------------------------------------------
    # Feature + modality logit extraction
    # ------------------------------------------------------------------

    def _extract_vectors_and_modality_logits(self, loader):
        """
        Extract backbone feature vectors and modality-specific logits z_m
        from the current network (eval mode).

        Returns:
            vectors:              np.ndarray [N, D]
            targets:              np.ndarray [N]
            modality_logits_dict: {modality: np.ndarray [N, C]}
        """
        self._network.eval()
        vectors, targets = [], []
        logits_dict = defaultdict(list)

        for _, _inputs, _targets in loader:
            for m in self._modality:
                _inputs[m] = _inputs[m].to(self._device)
            _targets = _targets.numpy()

            if isinstance(self._network, nn.DataParallel):
                _outputs = self._network.module.forward(_inputs)
            else:
                _outputs = self._network.forward(_inputs)

            _vectors = tensor2numpy(self._consensus(_outputs["features"]))
            vectors.append(_vectors)
            targets.append(_targets)

            for m in self._modality:
                _z_m = tensor2numpy(_outputs["auxiliary_logits"][m])
                logits_dict[m].append(_z_m)

        modality_logits = {
            m: np.concatenate(logits_dict[m], axis=0) if m in logits_dict else np.array([])
            for m in self._modality
        }
        return np.concatenate(vectors), np.concatenate(targets), modality_logits

    # ------------------------------------------------------------------
    # MoRST KD Loss   L_KD = Σ_m || z̃_m - z_m ||²
    # ------------------------------------------------------------------

    def _compute_morst_kd_loss(self, current_modality_logits: dict, stored_modality_logits: dict):
        """
        Compute modality-wise KD loss.

        L_KD = Σ_m β · || z̃_m - z_m ||²    (Eq. in paper, per modality sum)

        Args:
            current_modality_logits: {m: Tensor [B, C]}  current z_m
            stored_modality_logits:  {m: ndarray [B, C]}  z̃_m from replay buffer

        Returns:
            total_loss: Tensor scalar
        """
        total_loss = torch.tensor(0.0, device=self._device)

        if not isinstance(current_modality_logits, dict) or not isinstance(
            stored_modality_logits, dict
        ):
            return total_loss

        for m in self._modality:
            if m not in current_modality_logits or current_modality_logits[m] is None:
                continue
            if m not in stored_modality_logits or len(stored_modality_logits[m]) == 0:
                continue

            z_m = current_modality_logits[m]   # [B, C] Tensor
            z_tilde_m = stored_modality_logits[m]

            # Convert stored logits to tensor
            if isinstance(z_tilde_m, np.ndarray):
                z_tilde_m = torch.from_numpy(z_tilde_m).float().to(self._device)
            elif isinstance(z_tilde_m, torch.Tensor):
                z_tilde_m = z_tilde_m.float().to(self._device)
            else:
                continue

            if z_m.shape[0] != z_tilde_m.shape[0]:
                continue

            # Valid mask: skip padding entries (all -1)
            valid_mask = (z_tilde_m != -1).all(dim=1)
            if valid_mask.sum() == 0:
                continue

            # Ignore padded zero columns
            nonzero_mask = (z_tilde_m != 0).float()
            masked_z_m = z_m * nonzero_mask

            modality_loss = F.mse_loss(masked_z_m[valid_mask], z_tilde_m[valid_mask])
            total_loss = total_loss + self.morst_beta * modality_loss

        return total_loss

    # ------------------------------------------------------------------
    # Training loops
    # ------------------------------------------------------------------

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        Task 0 training — L_Sup only (no replay, no L_KD).

        L = L_Sup = L_CE(z_main, y) + λ · (1/|M|) Σ_m L_CE(z_m, y)
        """
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]

        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self._setup_epoch_and_collect_confidence(epoch)

            if self._partialbn:
                self._network.backbone.freeze_fn("partialbn_statistics")
            if self._freeze:
                self._network.backbone.freeze_fn("bn_statistics")

            losses = 0.0
            auxiliary_losses = 0.0
            correct = total = 0
            aux_correct = {m: 0 for m in self._modality}
            aux_total = {m: 0 for m in self._modality}

            for i, batch in enumerate(train_loader):
                if self.args["debug_mode"] and i >= 5:
                    break

                _, inputs, targets = batch
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)

                outputs = self._network(inputs, targets=targets)

                # L_Sup (includes λ * head supervision loss via mmeabase._compute_total_loss)
                loss_info = self._compute_total_loss(outputs, targets)
                total_loss = loss_info["total_loss"]
                auxiliary_loss = loss_info["auxiliary_loss"] * loss_info["aux_weight"]

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if self._clip_gradient is not None:
                    nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)
                for opt in optimizers:
                    opt.step()

                losses += total_loss.item()
                auxiliary_losses += auxiliary_loss.item()

                preds = torch.argmax(outputs["logits"], dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()

                aux_dict = outputs.get("auxiliary_logits", {})
                if isinstance(aux_dict, dict):
                    for m in self._modality:
                        if m in aux_dict and aux_dict[m] is not None:
                            aux_preds = torch.argmax(aux_dict[m], dim=1)
                            aux_correct[m] += aux_preds.eq(targets).sum().item()
                            aux_total[m] += targets.numel()

            for sch in schedulers:
                sch.step()

            train_acc = round((correct * 100.0) / max(1, total), 2)
            aux_acc_dict = {
                m: round((aux_correct[m] * 100.0) / aux_total[m], 2) if aux_total[m] > 0 else 0.0
                for m in self._modality
            }

            if self.args["use_wandb"]:
                wandb.log({
                    "Train/train_loss": losses / len(train_loader),
                    "Train/aux_loss": auxiliary_losses / len(train_loader),
                    "Train/train_accuracy": train_acc,
                    **{f"Train/aux_acc_{m}": aux_acc_dict[m] for m in self._modality},
                })

            aux_acc_str = ", ".join(
                [f"Aux_{m}_acc {aux_acc_dict[m]:.2f}" for m in self._modality]
            )
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{self._epochs} => "
                f"Loss {losses/len(train_loader):.3f}, "
                f"Aux_loss {auxiliary_losses/len(train_loader):.3f}, "
                f"Train_accy {train_acc:.2f}, {aux_acc_str}"
            )

            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                if self.args["use_wandb"]:
                    wandb.log({"Train/test_accuracy": test_acc})

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """
        Task t > 0 training — L_Sup + β · L_KD.

        L = L_Sup + β · L_KD
          = [L_CE(z_main,y) + λ·(1/|M|) Σ_m L_CE(z_m,y)]
            + β · Σ_m || z̃_m - z_m ||²
        """
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]

        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self._setup_epoch_and_collect_confidence(epoch)

            if self._partialbn:
                self._network.backbone.freeze_fn("partialbn_statistics")
            if self._freeze:
                self._network.backbone.freeze_fn("bn_statistics")

            losses = 0.0
            auxiliary_losses = 0.0
            morst_kd_losses = 0.0
            correct = total = 0
            aux_correct = {m: 0 for m in self._modality}
            aux_total = {m: 0 for m in self._modality}

            for i, batch in enumerate(train_loader):
                if self.args["debug_mode"] and i >= 5:
                    break

                # Batch from replay DataLoader includes stored modality logits z̃_m
                if len(batch) == 4:
                    _, inputs, targets, stored_logits_batch = batch
                else:
                    _, inputs, targets = batch
                    stored_logits_batch = None

                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)

                outputs = self._network(inputs, targets=targets)

                # L_Sup
                loss_info = self._compute_total_loss(outputs, targets)
                l_sup = loss_info["total_loss"]
                auxiliary_loss = loss_info["auxiliary_loss"] * loss_info["aux_weight"]

                # L_KD = β · Σ_m || z̃_m - z_m ||²
                if stored_logits_batch is not None:
                    if isinstance(stored_logits_batch, dict):
                        l_kd = self._compute_morst_kd_loss(
                            outputs.get("auxiliary_logits", {}), stored_logits_batch
                        )
                    else:
                        logging.warning(
                            "[MAND] stored_logits_batch is not a dict — skipping L_KD"
                        )
                        l_kd = torch.tensor(0.0, device=self._device)

                    total_loss = l_sup + l_kd
                    morst_kd_losses += l_kd.item()
                    auxiliary_losses += auxiliary_loss.item()
                else:
                    total_loss = l_sup
                    l_kd = torch.tensor(0.0, device=self._device)

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if self._clip_gradient is not None:
                    nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)
                for opt in optimizers:
                    opt.step()

                losses += total_loss.item()
                preds = torch.argmax(outputs["logits"], dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()

                aux_dict = outputs.get("auxiliary_logits", {})
                if isinstance(aux_dict, dict):
                    for m in self._modality:
                        if m in aux_dict and aux_dict[m] is not None:
                            aux_preds = torch.argmax(aux_dict[m], dim=1)
                            aux_correct[m] += aux_preds.eq(targets).sum().item()
                            aux_total[m] += targets.numel()

            for sch in schedulers:
                sch.step()

            train_acc = round((correct * 100.0) / max(1, total), 2)
            aux_acc_dict = {
                m: round((aux_correct[m] * 100.0) / aux_total[m], 2) if aux_total[m] > 0 else 0.0
                for m in self._modality
            }

            if self.args["use_wandb"]:
                wandb.log({
                    "Train/train_loss": losses / len(train_loader),
                    "Train/aux_loss": auxiliary_losses / len(train_loader),
                    "Train/morst_kd_loss": morst_kd_losses / len(train_loader),
                    "Train/train_accuracy": train_acc,
                    **{f"Train/aux_acc_{m}": aux_acc_dict[m] for m in self._modality},
                })

            aux_acc_str = ", ".join(
                [f"Aux_{m}_acc {aux_acc_dict[m]:.2f}" for m in self._modality]
            )
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{self._epochs} => "
                f"Loss {losses/len(train_loader):.3f}, "
                f"Aux_loss {auxiliary_losses/len(train_loader):.3f}, "
                f"MoRST_KD_loss {morst_kd_losses/len(train_loader):.3f}, "
                f"Train_accy {train_acc:.2f}, {aux_acc_str}"
            )

            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                if self.args["use_wandb"]:
                    wandb.log({"Train/test_accuracy": test_acc})

            prog_bar.set_description(info)

        logging.info(info)

    # ------------------------------------------------------------------
    # Class-wise confidence logging (diagnostic, same logic as MMEADER)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, weights_dir, filename):
        """Save model checkpoint to a parameter-specific subdirectory."""
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        if hasattr(self, "_class_means"):
            save_dict["class_means"] = self._class_means

        # kNN dist_ref: class-mean vectors + dist_stats
        fusion = getattr(self._network, "fusion", None) or getattr(self._network, "fusion_network", None)
        if fusion is not None and hasattr(fusion, "_class_means") and fusion._class_means:
            save_dict["class_means"] = fusion._class_means
            save_dict["dist_stats"]  = fusion._dist_stats
            if hasattr(fusion, "_raw_logit_arrays") and fusion._raw_logit_arrays:
                save_dict["raw_logit_arrays"] = fusion._raw_logit_arrays

        beta = self.morst_beta
        lambda_val = self.args.get("morst_lambda", self.args.get("aux_loss_weight", 0.5))
        param_subdir = f"beta{beta}_lambda{lambda_val}"

        weights_dir = os.path.join(weights_dir, param_subdir)
        os.makedirs(weights_dir, exist_ok=True)
        logging.info(f"[MAND] Saving checkpoint to: {param_subdir}")

        torch.save(save_dict, "{}/{}_{}.pkl".format(weights_dir, filename, self._cur_task))


# ------------------------------------------------------------------
# Concrete subclasses
# ------------------------------------------------------------------

class TBN_MAND(MAND):
    """MAND with TBN (Temporal Binding Network) backbone."""

    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)


class TSN_MAND(MAND):
    """MAND with TSN backbone."""

    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)
