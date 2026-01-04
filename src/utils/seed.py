"""
Reproducibility utilities.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility vs speed tradeoff
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    print(f"✓ Random seed set to {seed}")
