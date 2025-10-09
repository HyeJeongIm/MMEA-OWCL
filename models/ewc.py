import logging
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.mmeabase import MMEABaseLearner
from models.baseline_tbn import TBNBaseline
from models.baseline_tsn import TSNBaseline


# EWC hyperparameters
EPSILON = 1e-8
T = 2
lamda = 1000
fishermax = 0.0001


class EWC(MMEABaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
    def after_task(self):
        """Update known classes after task completion"""
        self._known_classes = self._total_classes
        
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
        """Main incremental training function"""
        self.total_classnum = data_manager.get_total_classnum()
        
        # Update task state
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        self._classes_seen_so_far = self._total_classes
        self.class_increments.append([self._known_classes, self._total_classes-1])
        
        # Update classifier for new classes
        self._update_classifier(self._total_classes)
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._total_classes-1))

        # Setup data loaders
        self._setup_data_loaders_with_ood(data_manager)

        # Multi-GPU setup
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Update Fisher information matrix
        if self.fisher is None:
            self.fisher = self.getFisherDiagonal(self.train_loader)
        else:
            alpha = self._known_classes / self._total_classes
            new_finsher = self.getFisherDiagonal(self.train_loader)
            for n, p in new_finsher.items():
                new_finsher[n][: len(self.fisher[n])] = (
                        alpha * self.fisher[n]
                        + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
                )
            self.fisher = new_finsher
        self.mean = {
            n: p.clone().detach()
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """Incremental training with EWC regularization"""
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]

        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            
            # 🔥 Fusion 모듈에 현재 epoch 정보 전달 (auxiliary_head_v2_4 warmup 지원)
            fusion_module = None
            if hasattr(self._network, 'fusion'):
                fusion_module = self._network.fusion
            elif hasattr(self._network, 'fusion_network'):
                fusion_module = self._network.fusion_network
            
            if fusion_module is not None and hasattr(fusion_module, 'set_epoch'):
                fusion_module.set_epoch(epoch)

            if self._partialbn:
                self._network.backbone.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.backbone.freeze()
            # if self._freeze:
            #     self._network.backbone.freeze_fn('bn_statistics')

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args["debug_mode"] and i >= 5:
                    break

                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                
                # 🎯 Forward pass with auxiliary loss support
                outputs = self._network(inputs, targets=targets)
                logits = outputs["logits"]
                
                # 🎯 Classification loss 계산 (EWC는 새 클래스만 대상)
                clf_targets = targets - self._known_classes
                clf_logits = outputs["logits"][:, self._known_classes:]
                
                # 🎯 Main loss를 clf_logits와 clf_targets로 재정의하여 auxiliary loss와 결합
                outputs_for_aux = {
                    'logits': clf_logits,
                    'auxiliary_loss': outputs.get('auxiliary_loss'),
                    'aux_loss_weight': outputs.get('aux_loss_weight')
                }
                loss_info = self._compute_total_loss(outputs_for_aux, clf_targets)
                loss_clf_with_aux = loss_info['total_loss']
                
                # 🎯 EWC regularization 추가
                loss_ewc = self.compute_ewc()
                loss = loss_clf_with_aux + lamda * loss_ewc

                # zero gradients
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)

                loss.backward()

                if self._clip_gradient is not None:
                    total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)

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

    def compute_ewc(self):
        """Compute EWC regularization loss"""
        loss = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.module.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
        else:
            for n, p in self._network.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
        return loss

    def getFisherDiagonal(self, train_loader):
        fisher = {
            n: torch.zeros(p.shape).to(self._device)
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }
        self._network.train()
        optimizer = optim.SGD(self._network.parameters(), lr=self._lr)
        for i, (_, inputs, targets) in enumerate(train_loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            targets = targets.to(self._device)
            logits = self._network(inputs)["logits"]
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher


class TBN_EWC(EWC):
    """MyEWC model with additional features for TBN"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)  # Assuming TBN is a custom network class
    

class TSN_EWC(EWC):
    """MyEWC model with additional features for TSN"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)  # Assuming TSN is a custom network class