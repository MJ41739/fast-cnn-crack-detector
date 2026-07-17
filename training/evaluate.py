import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from backend.config import settings
from training.dataset import get_dataloaders
from models.factory import get_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate_model(
    model_name: str, 
    model_path: str = None
) -> Dict[str, Any]:
    """Evaluate a trained model checkpoint on the test partition."""
    device = torch.device(settings.DEVICE)
    logger.info(f"Evaluating model: {model_name} on {device}")
    
    # Load dataset dataloaders
    _, _, test_loader = get_dataloaders()
    
    # Instantiate model and load best weights
    # Since we are evaluating the final model, freeze_backbone doesn't matter (we set to False to load all weights)
    model = get_model(model_name, pretrained=False, num_classes=1, freeze_backbone=False)
    
    if model_path is None:
        model_path = os.path.join(settings.CHECKPOINT_DIR, f"{model_name}_best.pth")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}. Did you train the model first?")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    # Measure latency
    latencies = []
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        
        # Warmup and time single image inference if batch_size=1 or compute average batch latency
        start_time = time.perf_counter()
        outputs = model(inputs)
        batch_latency = (time.perf_counter() - start_time) * 1000.0  # in ms
        
        # Approximate per-image latency
        latencies.append(batch_latency / inputs.size(0))
        
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(float)
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(labels.numpy().flatten())
        
    # Calculate performance metrics
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary")
    auc = roc_auc_score(all_targets, all_probs)
    avg_latency_ms = np.mean(latencies)
    
    logger.info(f"[{model_name.upper()} RESULTS]")
    logger.info(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | ROC-AUC: {auc:.4f}")
    logger.info(f"Average Inference Latency: {avg_latency_ms:.2f} ms/image")
    print(classification_report(all_targets, all_preds, target_names=["Negative (No Crack)", "Positive (Crack)"]))
    
    # Create evaluation metrics reports
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "avg_latency_ms": avg_latency_ms
    }
    
    # Save metrics JSON
    metrics_json_path = os.path.join(settings.OUTPUT_DIR, f"{model_name}_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plot and save confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, model_name)
    
    # Plot and save ROC Curve
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    plot_roc_curve(fpr, tpr, auc, model_name)
    
    return metrics

def plot_confusion_matrix(cm: np.ndarray, model_name: str):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Negative (No Crack)", "Positive (Crack)"],
        yticklabels=["Negative (No Crack)", "Positive (Crack)"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(settings.OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to: {save_path}")

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, model_name: str):
    """Plot and save ROC-AUC Curve."""
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(settings.OUTPUT_DIR, f"{model_name}_roc_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC-AUC curve saved to: {save_path}")

def compare_all_models():
    """Evaluate and compare all trained models, and save a comparison report."""
    models_to_evaluate = ["custom_cnn", "mobilenet_v2", "efficientnet", "rcnn"]
    results = {}
    
    for name in models_to_evaluate:
        weights_path = os.path.join(settings.CHECKPOINT_DIR, f"{name}_best.pth")
        if os.path.exists(weights_path):
            try:
                results[name] = evaluate_model(name, weights_path)
            except Exception as e:
                logger.error(f"Error evaluating model {name}: {e}")
        else:
            logger.warning(f"Skipping model {name} (checkpoint not found at {weights_path})")
            
    if not results:
        logger.error("No models were evaluated. Make sure checkpoints exist.")
        return
        
    # Write a summary report
    print("\n" + "="*80)
    print("MODEL COMPARISON ANALYSIS REPORT")
    print("="*80)
    
    # Find the best model based on weighted score: Accuracy (40%), F1-score (40%), Speed/Latency (20%)
    # Lower latency is better
    best_model_name = None
    best_score = -1.0
    
    comparison_table = []
    comparison_table.append("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Latency (ms) |")
    comparison_table.append("|-------|----------|-----------|--------|----------|---------|--------------|")
    
    for name, m in results.items():
        comparison_table.append(
            f"| {name} | {m['accuracy']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | "
            f"{m['f1_score']:.2%} | {m['roc_auc']:.4f} | {m['avg_latency_ms']:.2f} ms |"
        )
        
        # Normalized scores for selection
        # Latency penalty: 1 / (1 + latency)
        lat_score = 1.0 / (1.0 + (m['avg_latency_ms'] / 100.0))  # normalize latency to 0-1 range
        score = (m['accuracy'] * 0.4) + (m['f1_score'] * 0.4) + (lat_score * 0.2)
        
        if score > best_score:
            best_score = score
            best_model_name = name
            
    print("\n".join(comparison_table))
    print("="*80)
    print(f"RECOMMENDED BEST MODEL FOR PRODUCTION: {best_model_name.upper()}")
    print("="*80)
    
    # Save comparison markdown
    comparison_path = os.path.join(settings.OUTPUT_DIR, "model_comparison_report.md")
    with open(comparison_path, "w") as f:
        f.write("# Model Comparison Report\n\n")
        f.write("\n".join(comparison_table))
        f.write(f"\n\n**Recommended Best Model:** {best_model_name} (Score: {best_score:.4f})\n")
        
    # Update active model in local config files or JSON
    active_model_info = {
        "best_model": best_model_name,
        "accuracy": results[best_model_name]["accuracy"],
        "f1_score": results[best_model_name]["f1_score"],
        "latency_ms": results[best_model_name]["avg_latency_ms"]
    }
    with open(os.path.join(settings.CHECKPOINT_DIR, "best_model_info.json"), "w") as f:
        json.dump(active_model_info, f, indent=4)
        
    logger.info(f"Updated production config with best model metadata: {best_model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Crack Detection Models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="compare", 
        choices=["custom_cnn", "mobilenet_v2", "efficientnet", "rcnn", "compare"],
        help="Model to evaluate (or 'compare' to evaluate and compare all trained models)"
    )
    args = parser.parse_args()
    
    if args.model == "compare":
        compare_all_models()
    else:
        evaluate_model(args.model)
