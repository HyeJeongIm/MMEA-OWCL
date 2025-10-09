import logging
import copy

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb

from models.mmeabase import MMEABaseLearner
from utils.toolkit import tensor2numpy
from models.baseline_tbn import TBNBaseline
from models.baseline_tsn import TSNBaseline
EPSILON = 1e-8


class Replay(MMEABaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._num_segments = args["num_segments"]

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        # TSN: save parameters after task completion
        if hasattr(self._network, 'save_parameter'):
            self._network.save_parameter()
    
    def _update_classifier(self, nb_classes):
        """Update classifier based on network type"""
        if hasattr(self._network, 'update_fc'):
            # TBN: use update_fc method (total classes)
            self._network.update_fc(nb_classes)
        elif hasattr(self._network, 'gen_train_fc'):
            # TSN: use gen_train_fc method (incremental classes only)
            incre_classes = nb_classes - self._known_classes
            self._network.gen_train_fc(incre_classes)
        else:
            raise NotImplementedError(f"Network {type(self._network)} doesn't support classifier update")

    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._classes_seen_so_far = self._total_classes
        self.class_increments.append([self._known_classes, self._total_classes - 1])

        self._network.update_fc(self._total_classes)
        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")

        self._setup_data_loaders_with_ood(data_manager)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            
    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            for m in self._modality:
                _inputs[m] = _inputs[m].to(self._device)
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._consensus(self._network.module.extract_vector(_inputs))
                )
            else:
                _vectors = tensor2numpy(
                    self._consensus(self._network.extract_vector(_inputs))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _consensus(self, x):
        output = x.view((-1, self._num_segments) + x.size()[1:])
        output = output.mean(dim=1, keepdim=True)
        output = output.squeeze(1)
        return output

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean
            
    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]

            m = min(m, vectors.shape[0])
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                # print(mu_p)
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    data[i]
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    vectors[i]
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
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

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            
            self._class_means[class_idx, :] = mean

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """Enhanced Replay with Auxiliary Head Training"""
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]
        
        # Check if using adaptive fusion
        use_adaptive_fusion = hasattr(self._network.fusion, 'compute_synergy_metrics')
        aux_loss_weight = self.args.get("aux_loss_weight", 0.3)  # auxiliary loss 가중치

        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            if self._partialbn:
                self._network.backbone.freeze_fn("partialbn_statistics")
            if self._freeze:
                self._network.backbone.freeze_fn("bn_statistics")

            losses, aux_losses, correct, total = 0.0, 0.0, 0, 0
            synergy_stats = {}
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args["debug_mode"] and i >= 5:
                    break

                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)

                # Forward pass with auxiliary head support
                if use_adaptive_fusion:
                    outputs = self._network(inputs, targets=targets, return_aux=True)
                    logits = outputs["logits"]
                    main_loss = F.cross_entropy(logits, targets)
                    
                    # Auxiliary head losses
                    aux_loss = 0.0
                    if 'aux_logits' in outputs and outputs['aux_logits']:
                        for mod, aux_logit in outputs['aux_logits'].items():
                            aux_loss += F.cross_entropy(aux_logit, targets)
                        aux_loss /= len(outputs['aux_logits'])  # 평균
                    
                    # Total loss = main loss + weighted auxiliary loss
                    total_loss = main_loss + aux_loss_weight * aux_loss
                    
                    # Synergy information 수집
                    if 'synergy_info' in outputs:
                        for key, value in outputs['synergy_info'].items():
                            if key not in synergy_stats:
                                synergy_stats[key] = []
                            synergy_stats[key].append(value)
                else:
                    # 기존 방식
                    outputs = self._network(inputs)
                    logits = outputs["logits"]
                    total_loss = F.cross_entropy(logits, targets)
                    aux_loss = torch.tensor(0.0)

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if self._clip_gradient is not None:
                    nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)
                for opt in optimizers:
                    opt.step()

                losses += total_loss.item()
                aux_losses += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()

            for sch in schedulers:
                sch.step()

            train_acc = round((correct * 100.0) / max(1, total), 2)
            
            # Synergy statistics 계산
            avg_synergy_stats = {}
            for key, values in synergy_stats.items():
                if values:  # 빈 리스트가 아닌 경우만
                    avg_synergy_stats[f"avg_{key}"] = np.mean(values)

            # wandb 로깅
            log_dict = {
                "Train/train_loss": losses / len(train_loader),
                "Train/train_accuracy": train_acc,
            }
            
            if use_adaptive_fusion:
                log_dict["Train/aux_loss"] = aux_losses / len(train_loader)
                log_dict.update({f"Train/{k}": v for k, v in avg_synergy_stats.items()})
            
            if self.args["use_wandb"]:
                wandb.log(log_dict)

            info = f"Task {self._cur_task}, Epoch {epoch+1}/{self._epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            
            if use_adaptive_fusion:
                info += f", Aux_loss {aux_losses/len(train_loader):.3f}"
                
                # 주요 시너지 정보 표시
                if 'avg_Gyro_synergy_score' in avg_synergy_stats:
                    info += f", Gyro_synergy {avg_synergy_stats['avg_Gyro_synergy_score']:.3f}"
                if 'avg_Acce_synergy_score' in avg_synergy_stats:
                    info += f", Acce_synergy {avg_synergy_stats['avg_Acce_synergy_score']:.3f}"
            
            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                if self.args["use_wandb"]:
                    wandb.log({"Train/test_accuracy": test_acc})

            prog_bar.set_description(info)

        logging.info(info)
        
        # 에포크 종료 후 시너지 요약 정보 로깅
        if use_adaptive_fusion and hasattr(self._network.fusion, 'get_synergy_summary'):
            synergy_summary = self._network.fusion.get_synergy_summary()
            if synergy_summary:
                logging.info("=== Modality Synergy Summary ===")
                for mod, trend in synergy_summary.items():
                    if 'trend' in mod:
                        logging.info(f"  {mod}: {trend}")
                logging.info("=" * 35)


class TBN_Replay(Replay):
    """MyReplay model with additional features for TBN"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)  # Assuming TBN is a custom network class
    

class TSN_Replay(Replay):
    """MyReplay model with additional features for TSN"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)  # Assuming TSN is a custom network class

