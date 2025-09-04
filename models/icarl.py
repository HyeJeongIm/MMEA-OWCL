import logging
import copy
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from utils.toolkit import tensor2numpy
from models.mmeabase import MMEABaseLearner
from models.baseline_tbn import TBNBaseline
from models.baseline_tsn import TSNBaseline


EPSILON = 1e-8
T = 2

class iCaRL(MMEABaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
        self._num_segments = args["num_segments"]

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
 # TBN: save parameters after task completion
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
        self.class_increments.append([self._known_classes, self._total_classes-1])
        
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # Setup data loaders
        self._setup_data_loaders_with_ood(data_manager)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]

        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            
            if self._partialbn:
                self._network.backbone.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.backbone.freeze()
                
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args["debug_mode"] and i >= 5:
                    break
                
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = loss_clf + loss_kd

                # zero gradients
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                    
                loss.backward()
                
                if self._clip_gradient is not None:
                    total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)
                    # if total_norm > self._clip_gradient:
                    #     print("clipping gradient: {} with coef {}".format(total_norm, self._clip_gradient / total_norm))
                        
                # optimizer step
                for opt in optimizers:
                    opt.step()
                    
                losses += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()

            # epoch-level scheduler step
            for sch in schedulers:
                sch.step()
                
            train_acc = round((correct * 100.0) / max(1, total), 2)

            # Log training metrics to W&B
            if self.args['use_wandb']:
                wandb.log({
                    "Train/train_loss": losses / len(train_loader),
                    "Train/train_accuracy": train_acc
                })

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self._epochs,
                losses / len(train_loader),
                train_acc,
            )
            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                # Log test metrics to W&B
                if self.args['use_wandb']:
                    wandb.log({
                        "Train/test_accuracy": test_acc
                    })
            
            prog_bar.set_description(info)
        logging.info(info)

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


class TBN_iCaRL(iCaRL):
    """MyiCaRL model with additional features for TBN"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)  # Assuming TBN is a custom network class
    

class TSN_iCaRL(iCaRL):
    """MyiCaRL model with additional features for TSN"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)  # Assuming TSN is a custom network class


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
