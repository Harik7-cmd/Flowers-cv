"""
Visualization utilities for analysis and presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(history, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, [a*100 for a in history['train_acc']], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, [a*100 for a in history['val_acc']], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return cm


def plot_class_distribution(labels, class_names=None, save_path=None):
    """
    Plot distribution of samples per class.
    """
    from collections import Counter
    counts = Counter(labels)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(counts)), [counts[i] for i in range(len(counts))], color='steelblue')
    ax.axhline(y=np.mean(list(counts.values())), color='red', linestyle='--', label='Mean')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
