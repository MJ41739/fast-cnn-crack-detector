import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Crack Detection System"
    DEBUG: bool = False
    
    # Dataset Settings
    DATASET_DIR: str = Field(
        default=str(BASE_DIR / "kaggle_dataset"),
        description="Path to the kaggle dataset containing Positive and Negative folders"
    )
    TRAIN_SPLIT: float = 0.70
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    
    # Model Hyperparameters
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 0  # Set to 0 if encountering issues on Windows
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 15
    SEED: int = 42
    
    # Checkpoints and Outputs
    CHECKPOINT_DIR: str = str(BASE_DIR / "checkpoints")
    OUTPUT_DIR: str = str(BASE_DIR / "outputs")
    
    # Active Model in production API (can be 'custom_cnn', 'mobilenet_v2', 'efficientnet', 'rcnn', etc.)
    ACTIVE_MODEL_NAME: str = "custom_cnn"
    
    # Device setup: automatically default to cuda if available
    DEVICE: str = "cuda"  # Will be validated to fall back to 'cpu' if cuda is not available
    
    # Model Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings
settings = Settings()

# Post-processing to create directories and validate device
os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

import torch
if settings.DEVICE == "cuda" and not torch.cuda.is_available():
    settings.DEVICE = "cpu"
