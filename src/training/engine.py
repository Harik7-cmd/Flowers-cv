"""
Training and evaluation loops with mixed precision support.
"""

import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, use_amp=True):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        loader: Training DataLoader
        optimizer: Optimizer
        loss_fn: Loss function
        scaler: GradScaler for mixed precision
        device: Device (cuda/cpu)
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.1f}%"})
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp=True):
    """
    Evaluate model on a dataset.
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        with autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def get_predictions(model, loader, device, use_amp=True):
    """
    Get all predictions from a model.
    
    Returns:
        predictions (np.array), labels (np.array), probabilities (np.array)
    """
    import torch.nn.functional as F
    import numpy as np
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(device)
        
        with autocast('cuda', enabled=use_amp):
            outputs = model(images)
        
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)
