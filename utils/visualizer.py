import os
import matplotlib.pyplot as plt
from typing import Dict, List
from backend.config import settings

def plot_training_history(history: Dict[str, List[float]], model_name: str, save_path: str = None):
    """Plot training and validation accuracy and loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Loss Plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#3b82f6", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ef4444", linewidth=2)
    ax1.set_title(f"Training & Validation Loss ({model_name})", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # 2. Accuracy Plot
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#10b981", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "s-", label="Val Acc", color="#f59e0b", linewidth=2)
    ax2.set_title(f"Training & Validation Accuracy ({model_name})", fontsize=14, fontweight="bold", pad=10)
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.legend(frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    
    # Save target directory
    if save_path is None:
        save_path = os.path.join(settings.OUTPUT_DIR, f"{model_name}_training_history.png")
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {save_path}")

if __name__ == "__main__":
    # Test plot
    dummy_history = {
        "train_loss": [0.6, 0.4, 0.2, 0.1],
        "val_loss": [0.5, 0.3, 0.25, 0.15],
        "train_acc": [70.0, 85.0, 92.0, 96.0],
        "val_acc": [75.0, 84.0, 90.0, 95.0]
    }
    plot_training_history(dummy_history, "test_model")
