import logging
import torch
import torch.nn as nn
import numpy as np

from models.mmeabase import MMEABaseLearner
from models.baseline_tbn import TBNBaseline
from models.baseline_tsn import TSNBaseline

class UpperBound(MMEABaseLearner):
    """
    TBN Upper-Bound Model for comprehensive performance analysis
    - Trains on all classes simultaneously (no incremental learning)
    - Generates confusion matrix and class-wise analysis
    - Provides upper-bound performance baseline
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        logging.info("🎯 Upper-Bound Model initialized")
        logging.info(f"   📊 Total Classes: {args.get('init_cls', args.get('increment', 32))}")
        logging.info(f"   🔧 Device: {self._device}")
        
    def after_task(self):
        """Update known classes after task completion"""
        self._known_classes = self._total_classes
        
    def _update_classifier(self, nb_classes):
        """Update classifier for all classes at once"""
        self._network.update_fc(nb_classes)
        
    def incremental_train(self, data_manager):
        """Main training function for upper-bound (train on all classes at once)"""
        self.total_classnum = data_manager.get_total_classnum()
        
        # Set up for upper-bound: single task with all classes
        self._cur_task = 0  # Only one task
        self._total_classes = self.total_classnum  # All classes
        self._known_classes = 0  # Start from scratch
        self._classes_seen_so_far = self._total_classes
        self.class_increments = [[0, self._total_classes - 1]]
        
        # Update classifier for all classes
        self._update_classifier(self._total_classes)
        logging.info(f"🎯 Upper-bound: Training on ALL classes 0-{self._total_classes-1}")
        
        # Setup data loaders using base method but for all classes
        self._setup_data_loaders_with_ood(data_manager)
        
        # Multi-GPU setup
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        # Train the model using base train method
        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module



class TBN_UpperBound(UpperBound):
    """TBN Upper-Bound Model for comprehensive performance analysis"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)
        
        logging.info("🎯 TBN Upper-Bound Model initialized")
        logging.info(f"   🏗️  Network: TBNBaseline")
        logging.info(f"   📊 Modalities: {self._modality}")
        logging.info(f"   🎨 Fusion: {args.get('fusion_type', 'concat')}")
        logging.info(f"   📈 Epochs: {self._epochs}")
        logging.info(f"   🔧 Device: {self._device}")
        
class TSN_UpperBound(UpperBound):
    """TBN Upper-Bound Model for comprehensive performance analysis"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)
        
        logging.info("🎯 TBN Upper-Bound Model initialized")
        logging.info(f"   🏗️  Network: TBNBaseline")
        logging.info(f"   📊 Modalities: {self._modality}")
        logging.info(f"   🎨 Fusion: {args.get('fusion_type', 'concat')}")
        logging.info(f"   📈 Epochs: {self._epochs}")
        logging.info(f"   🔧 Device: {self._device}")

