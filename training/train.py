import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Any

from backend.config import settings
from training.dataset import get_dataloaders
from models.factory import get_model
from utils.visualizer import plot_training_history

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def train_epoch(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    scaler: GradScaler, 
    device: torch.device
) -> tuple[float, float]:
    """Train the model for one epoch using mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Mixed Precision context
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backpropagation with GradScaler
        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        
        # Apply sigmoid to outputs for prediction thresholding
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate_epoch(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> tuple[float, float]:
    """Validate the model on validation data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        running_loss += loss.item() * inputs.size(0)
        
        # Compute accuracy
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def train_model(
    model_name: str, 
    resume: bool = False, 
    epochs: int = None, 
    batch_size: int = None
) -> Dict[str, Any]:
    """Main training loop for a specific model configuration."""
    # Override settings if requested
    if epochs:
        settings.NUM_EPOCHS = epochs
    if batch_size:
        settings.BATCH_SIZE = batch_size
        
    device = torch.device(settings.DEVICE)
    logger.info(f"Training model: '{model_name}' on device: {device}")
    
    # Load dataset dataloaders
    train_loader, val_loader, _ = get_dataloaders()
    
    # Load model from factory
    # If using custom CNN, we train from scratch. If MobileNet/EfficientNet, we load pretrained.
    # We will do a two-stage training approach:
    # Stage 1: Freeze backbone, train classification head for 3 epochs.
    # Stage 2: Unfreeze backbone, fine-tune the entire network with a small learning rate.
    is_transfer = model_name in ["mobilenet_v2", "efficientnet"]
    
    if is_transfer:
        logger.info(f"Setting up transfer learning for {model_name} (Stage 1: frozen backbone)")
        model = get_model(model_name, pretrained=True, freeze_backbone=True)
    else:
        logger.info(f"Initializing {model_name} from scratch")
        model = get_model(model_name, pretrained=False, freeze_backbone=False)
        
    model = model.to(device)
    
    # Criterion & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    # Learning rate settings
    lr = settings.LEARNING_RATE
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    
    checkpoint_path = os.path.join(settings.CHECKPOINT_DIR, f"{model_name}_latest.pth")
    best_model_path = os.path.join(settings.CHECKPOINT_DIR, f"{model_name}_best.pth")
    
    # Resume training if checkpoint exists and requested
    if resume and os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Re-initialize the model with unfrozen layers if it was unfrozen during the checkpoint epoch
        if checkpoint.get("unfrozen", False) and is_transfer:
            model = get_model(model_name, pretrained=False, freeze_backbone=False)
            model = model.to(device)
            # Recreate optimizer to update all parameters
            optimizer = optim.AdamW(model.parameters(), lr=checkpoint.get("lr", lr), weight_decay=1e-2)
            
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    stage2_triggered = False
    
    # Run training loop
    for epoch in range(start_epoch, settings.NUM_EPOCHS):
        # Transfer learning phase management: unfreeze backbone at epoch 3
        if is_transfer and epoch >= 3 and not stage2_triggered:
            # Check if checkpoint already unfroze it
            unfreeze = True
            if resume and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if checkpoint.get("unfrozen", False):
                    unfreeze = False
            
            if unfreeze:
                logger.info("Stage 2: Unfreezing backbone for fine-tuning...")
                # We reload model architecture with unfrozen layers but preserve the state dict of the classification head
                state_dict = model.state_dict()
                model = get_model(model_name, pretrained=False, freeze_backbone=False)
                model.load_state_dict(state_dict)
                model = model.to(device)
                
                # Lower learning rate for fine tuning
                lr = settings.LEARNING_RATE * 0.1
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
                stage2_triggered = True
                
        epoch_start_time = time.time()
        
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        logger.info(
            f"Epoch [{epoch+1}/{settings.NUM_EPOCHS}] ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_loss": best_val_loss,
            "unfrozen": (is_transfer and epoch >= 3) or not is_transfer,
            "lr": lr
        }, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with Val Loss: {val_loss:.4f}")
            
        # Check early stopping
        if early_stopping(val_loss):
            logger.info("Early stopping triggered. Training stopped.")
            break
            
    # Load best weights before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("Loaded best model weights for final visualization.")
        
    # Plot training curves
    plot_training_history(history, model_name)
    
    # Save training history metadata
    history_json_path = os.path.join(settings.OUTPUT_DIR, f"{model_name}_history.json")
    with open(history_json_path, "w") as f:
        json.dump(history, f, indent=4)
        
    return {
        "model_name": model_name,
        "best_val_loss": best_val_loss,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
        "history": history
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crack Detection Models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="custom_cnn", 
        choices=["custom_cnn", "mobilenet_v2", "efficientnet", "all"],
        help="Model to train (or 'all' to train and compare all models)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    
    args = parser.parse_args()
    
    # Ensure dataset is present
    if not os.path.exists(settings.DATASET_DIR):
        print(f"Error: Dataset directory {settings.DATASET_DIR} does not exist.")
        exit(1)
        
    if args.model == "all":
        results = {}
        for m_name in ["custom_cnn", "mobilenet_v2", "efficientnet"]:
            print("="*60)
            print(f"TRAINING {m_name.upper()}")
            print("="*60)
            results[m_name] = train_model(m_name, resume=args.resume, epochs=args.epochs, batch_size=args.batch_size)
        
        # Summarize results
        print("\n" + "="*40 + "\nTRAINING COMPARISON SUMMARY\n" + "="*40)
        for name, res in results.items():
            print(f"Model: {name} | Best Val Loss: {res['best_val_loss']:.4f} | Final Val Acc: {res['final_val_acc']:.2f}%")
    else:
        train_model(args.model, resume=args.resume, epochs=args.epochs, batch_size=args.batch_size)
