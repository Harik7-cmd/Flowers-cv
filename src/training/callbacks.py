"""
Training callbacks for early stopping and checkpointing.
"""

import torch


class EarlyStopping:
    """
    Stop training when validation metric stops improving.
    
    Args:
        patience: Epochs to wait before stopping
        mode: 'min' for loss, 'max' for accuracy
        delta: Minimum change to qualify as improvement
    """
    
    def __init__(self, patience: int = 7, mode: str = 'max', delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, metric):
        if self.mode == 'max':
            score = metric
            improved = self.best_score is None or score > self.best_score + self.delta
        else:
            score = -metric
            improved = self.best_score is None or score > self.best_score + self.delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # No improvement


class ModelCheckpoint:
    """
    Save model when validation metric improves.
    
    Args:
        filepath: Path to save checkpoint
        mode: 'min' for loss, 'max' for accuracy
    """
    
    def __init__(self, filepath: str, mode: str = 'max'):
        self.filepath = filepath
        self.mode = mode
        self.best_score = None
    
    def __call__(self, metric, model, optimizer=None, epoch=None, **kwargs):
        if self.mode == 'max':
            improved = self.best_score is None or metric > self.best_score
        else:
            improved = self.best_score is None or metric < self.best_score
        
        if improved:
            self.best_score = metric
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'val_metric': metric,
                'epoch': epoch,
                **kwargs
            }
            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(checkpoint, self.filepath)
            return True
        return False
