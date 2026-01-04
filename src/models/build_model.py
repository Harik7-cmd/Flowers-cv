"""
Model building utilities using timm library.
"""

import torch
import timm


def build_model(model_name: str, num_classes: int = 102, pretrained: bool = True):
    """
    Build a model using timm library.
    
    Args:
        model_name: timm model identifier
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        PyTorch model
    
    Recommended models for flowers:
        - 'tf_efficientnet_b2.ns_jft_in1k': Good accuracy/speed tradeoff
        - 'vit_small_patch16_224.augreg_in21k_ft_in1k': Best accuracy
        - 'resnet50.a1_in1k': Strong baseline
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def freeze_backbone(model):
    """Freeze all parameters except the classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze head (works for most timm models)
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model


def unfreeze_all(model):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_parameter_groups(model, lr_backbone: float = 1e-5, lr_head: float = 1e-4):
    """
    Get parameter groups with differential learning rates.
    
    Args:
        model: The model
        lr_backbone: Learning rate for pretrained layers
        lr_head: Learning rate for classifier head
    
    Returns:
        List of parameter groups for optimizer
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "head" in name or "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head}
    ]


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
