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


T = 2
lamda = 1


class LwF(MMEABaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
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
        """Main incremental training function"""
        self.total_classnum = data_manager.get_total_classnum()
        
        # Update task state
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        self._classes_seen_so_far = self._total_classes
        self.class_increments.append([self._known_classes, self._total_classes-1])
        
        # Update classifier for new classes
        self._network.update_fc(self._total_classes)
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._total_classes-1))

        # Setup data loaders
        self._setup_data_loaders_with_ood(data_manager)

        # Multi-GPU setup
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """Incremental training with EWC regularization"""
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]

        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            if self._partialbn:
                self._network.backbone.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.backbone.freeze_fn('bn_statistics')

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args["debug_mode"] and i >= 5:
                    break

                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                logits = self._network(inputs)["logits"]

                # Classification loss + LwF regularization
                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = lamda * loss_kd + loss_clf

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

           
class TBN_LwF(LwF):
    """MyLwF model with additional features for TBN"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)  # Assuming TBN is a custom network class
    

class TSN_LwF(LwF):
    """MyLwF model with additional features for TSN"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)  # Assuming TSN is a custom network class


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]