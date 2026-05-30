import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import logging
from typing import Tuple, List, Dict
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from backend.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AddGaussianNoise(object):
    """Custom transform to add Gaussian noise to a tensor image."""
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

class CrackDataset(Dataset):
    """Custom PyTorch dataset for crack/non-crack images."""
    def __init__(self, file_paths: List[str], labels: List[int], transform = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Open image in RGB mode
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img_tensor = self.transform(img)
                else:
                    # Basic fallback transform if none provided
                    basic_transform = transforms.Compose([
                        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
                        transforms.ToTensor(),
                    ])
                    img_tensor = basic_transform(img)
            
            return img_tensor, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}. Swapping with a random valid image.")
            # Resolve corruption by returning a random valid image
            random_idx = random.randint(0, len(self.file_paths) - 1)
            # Avoid infinite recursion by choosing an index we know has not failed yet
            if random_idx == idx:
                random_idx = (idx + 1) % len(self.file_paths)
            return self.__getitem__(random_idx)

def get_data_splits() -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """Scan dataset directories and partition into train, val, and test splits."""
    positive_dir = os.path.join(settings.DATASET_DIR, "Positive")
    negative_dir = os.path.join(settings.DATASET_DIR, "Negative")
    
    if not os.path.exists(positive_dir) or not os.path.exists(negative_dir):
        raise FileNotFoundError(
            f"Dataset folders 'Positive' or 'Negative' not found in: {settings.DATASET_DIR}. "
            "Please check config or set DATASET_DIR correctly."
        )

    # Gather positive images
    pos_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    pos_labels = [1] * len(pos_files)

    # Gather negative images
    neg_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    neg_labels = [0] * len(neg_files)
    
    logger.info(f"Dataset found: {len(pos_files)} Positive images, {len(neg_files)} Negative images.")

    # Combine datasets
    all_files = pos_files + neg_files
    all_labels = pos_labels + neg_labels
    
    # Class balance check
    pos_count = sum(all_labels)
    neg_count = len(all_labels) - pos_count
    logger.info(f"Class distribution: Positive = {pos_count} ({pos_count/len(all_labels):.1%}), Negative = {neg_count} ({neg_count/len(all_labels):.1%})")

    # Set seed for reproducible splits
    random.seed(settings.SEED)
    np.random.seed(settings.SEED)
    
    # First split: Train vs Remaining (Val + Test)
    val_test_size = settings.VAL_SPLIT + settings.TEST_SPLIT
    train_files, val_test_files, train_labels, val_test_labels = train_test_split(
        all_files, all_labels, 
        test_size=val_test_size, 
        stratify=all_labels, 
        random_state=settings.SEED
    )
    
    # Second split: Val vs Test (from remaining split)
    val_ratio = settings.VAL_SPLIT / val_test_size
    val_files, test_files, val_labels, test_labels = train_test_split(
        val_test_files, val_test_labels, 
        test_size=(1.0 - val_ratio), 
        stratify=val_test_labels, 
        random_state=settings.SEED
    )
    
    logger.info(f"Split sizes: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    return train_files, val_files, test_files, train_labels, val_labels, test_labels

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Define data augmentation transforms for training and standard normalization for validation/testing."""
    
    # Mean and Std for ImageNet normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        # Brightness, contrast and saturation adjustments
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        # Normalize to PyTorch ImageNet standards
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        # Noise injection after converting to Tensor
        AddGaussianNoise(mean=0.0, std=0.03)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, val_test_transform

def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Main wrapper function to build train, val, and test DataLoaders."""
    train_files, val_files, test_files, train_labels, val_labels, test_labels = get_data_splits()
    train_trans, val_test_trans = get_transforms()
    
    train_dataset = CrackDataset(train_files, train_labels, transform=train_trans)
    val_dataset = CrackDataset(val_files, val_labels, transform=val_test_trans)
    test_dataset = CrackDataset(test_files, test_labels, transform=val_test_trans)
    
    # Pin memory is useful for fast transfer to GPU VRAM
    # For Windows, sometimes num_workers > 0 causes issues in notebooks, but it works fine in main scripts.
    # In case of problems, users can set num_workers to 0 in config.py.
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=settings.NUM_WORKERS,
        pin_memory=True if settings.DEVICE == "cuda" else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        num_workers=settings.NUM_WORKERS,
        pin_memory=True if settings.DEVICE == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        num_workers=settings.NUM_WORKERS,
        pin_memory=True if settings.DEVICE == "cuda" else False
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test split function
    try:
        tr, vl, ts = get_dataloaders()
        print("DataLoaders instantiated successfully.")
        print(f"Train batches: {len(tr)}, Val batches: {len(vl)}, Test batches: {len(ts)}")
    except Exception as e:
        print(f"Error during Dataloader initialization: {e}")
