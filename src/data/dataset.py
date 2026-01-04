"""
Dataset classes and data loading utilities for Oxford 102 Flowers.
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Flowers102
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics (required for pretrained models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Official Oxford 102 flower names
FLOWER_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower",
    "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus",
    "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress",
    "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
]


class FlowersDataset(Dataset):
    """
    Wrapper for Oxford 102 Flowers dataset with albumentations support.
    
    Args:
        root: Data directory path
        split: 'train', 'val', or 'test'
        transform: Albumentations transform pipeline
        download: Whether to download if not present
    """
    
    def __init__(self, root: str, split: str, transform=None, download: bool = True):
        self.dataset = Flowers102(root=root, split=split, download=download)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)  # PIL to numpy for albumentations
        
        if self.transform:
            img = self.transform(image=img)["image"]
        
        return img, label


def get_train_transforms(img_size: int = 224):
    """
    Training augmentations optimized for flower classification.
    
    Includes:
    - RandomResizedCrop: Handle different flower sizes
    - HorizontalFlip: Flowers are symmetric
    - Affine: Rotation and scale variations
    - ColorJitter: Critical for color-discriminative flowers
    """
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.15), rotate=(-30, 30), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.7),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_eval_transforms(img_size: int = 224):
    """
    Evaluation transforms (no augmentation, deterministic).
    """
    return A.Compose([
        A.Resize(height=int(img_size * 1.15), width=int(img_size * 1.15)),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_dataloaders(data_root: str, batch_size: int = 32, num_workers: int = 4, img_size: int = 224):
    """
    Create train, validation, and test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_train_transforms(img_size)
    eval_transform = get_eval_transforms(img_size)
    
    train_dataset = FlowersDataset(data_root, "train", train_transform)
    val_dataset = FlowersDataset(data_root, "val", eval_transform)
    test_dataset = FlowersDataset(data_root, "test", eval_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
