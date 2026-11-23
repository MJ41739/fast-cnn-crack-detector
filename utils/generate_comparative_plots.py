import os
import sys
import json
import time
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, roc_curve, 
    precision_recall_curve, auc, matthews_corrcoef, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings, BASE_DIR
from training.dataset import get_dataloaders
from models.factory import get_model

# Set seaborn style for publication quality figures
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11})

def compute_all_metrics():
    """Evaluate all available models and calculate exhaustive evaluation metrics."""
    device = torch.device(settings.DEVICE)
    models_to_evaluate = ["custom_cnn", "mobilenet_v2", "rcnn", "efficientnet"]
    
    print("Loading test dataloader for comprehensive metric calculation...", flush=True)
    _, _, test_loader = get_dataloaders()
    
    comprehensive_metrics = {}
    curves_data = {}

    for name in models_to_evaluate:
        weights_path = os.path.join(settings.CHECKPOINT_DIR, f"{name}_best.pth")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(settings.CHECKPOINT_DIR, f"{name}_latest.pth")
        
        if not os.path.exists(weights_path):
            print(f"Skipping model '{name}': No weight checkpoint found at {weights_path}", flush=True)
            continue

        print(f"Evaluating model '{name}' from checkpoint '{weights_path}'...", flush=True)
        try:
            model = get_model(name, pretrained=False, num_classes=1, freeze_backbone=False)
            state_dict = torch.load(weights_path, map_location=device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()

            all_preds = []
            all_probs = []
            all_targets = []
            latencies = []

            for b_idx, (inputs, labels) in enumerate(test_loader, 1):
                inputs = inputs.to(device)
                
                t0 = time.perf_counter()
                with torch.no_grad():
                    outputs = model(inputs)
                batch_ms = (time.perf_counter() - t0) * 1000.0

                latencies.append(batch_ms / inputs.size(0))

                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(float)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(labels.numpy().flatten())

            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)

            # Confusion matrix metrics
            cm = confusion_matrix(all_targets, all_preds)
            tn, fp, fn, tp = cm.ravel()

            acc = accuracy_score(all_targets, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            roc_auc = roc_auc_score(all_targets, all_probs)

            # Precision-Recall AUC
            prec_curve, rec_curve, _ = precision_recall_curve(all_targets, all_probs)
            pr_auc = auc(rec_curve, prec_curve)

            # Matthews Correlation Coefficient & Log Loss
            mcc = matthews_corrcoef(all_targets, all_preds)
            bce_loss = log_loss(all_targets, all_probs)

            avg_latency = float(np.mean(latencies))
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

            # Parameters and File Size
            num_params = sum(p.numel() for p in model.parameters())
            file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

            comprehensive_metrics[name] = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall_sensitivity": float(recall),
                "specificity": float(specificity),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "mcc": float(mcc),
                "log_loss": float(bce_loss),
                "latency_ms": avg_latency,
                "fps": float(fps),
                "num_params": num_params,
                "file_size_mb": float(file_size_mb),
                "confusion_matrix": {
                    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
                }
            }

            # Curves data for combined plotting
            fpr, tpr, _ = roc_curve(all_targets, all_probs)
            curves_data[name] = {
                "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
                "rec_curve": rec_curve, "prec_curve": prec_curve, "pr_auc": pr_auc
            }

            print(f"Model '{name}' evaluated successfully! Acc: {acc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}, Latency: {avg_latency:.2f}ms", flush=True)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}", flush=True)

    # Save comprehensive metrics JSON
    json_path = os.path.join(settings.OUTPUT_DIR, "comprehensive_metrics_comparison.json")
    with open(json_path, "w") as f:
        json.dump(comprehensive_metrics, f, indent=4)
    print(f"Saved comprehensive metrics JSON to: {json_path}", flush=True)

    return comprehensive_metrics, curves_data

def generate_all_plots(metrics: dict, curves_data: dict):
    """Generate all high-resolution comparative figures."""
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    model_colors = {
        "custom_cnn": "#2563eb",     # Vibrant Blue
        "mobilenet_v2": "#10b981",   # Emerald Green
        "rcnn": "#8b5cf6",           # Purple
        "efficientnet": "#f59e0b"    # Amber
    }
    model_display_names = {
        "custom_cnn": "FASTCNN (Custom CNN)",
        "mobilenet_v2": "MobileNetV2",
        "rcnn": "Recurrent CNN (RCNN)",
        "efficientnet": "EfficientNet-B0"
    }

    # 1. Combined ROC Curves Plot
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    for name, c_data in curves_data.items():
        color = model_colors.get(name, "#333333")
        label = f"{model_display_names.get(name, name)} (AUC = {c_data['roc_auc']:.4f})"
        ax.plot(c_data["fpr"], c_data["tpr"], lw=2.5, color=color, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Baseline (AUC = 0.50)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) - Multi-Model Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_vs_path = os.path.join(settings.OUTPUT_DIR, "combined_roc_curves_vs.png")
    plt.savefig(roc_vs_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {roc_vs_path}", flush=True)

    # 2. Combined Precision-Recall Curves Plot
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    for name, c_data in curves_data.items():
        color = model_colors.get(name, "#333333")
        label = f"{model_display_names.get(name, name)} (PR-AUC = {c_data['pr_auc']:.4f})"
        ax.plot(c_data["rec_curve"], c_data["prec_curve"], lw=2.5, color=color, label=label)

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    ax.set_title("Precision-Recall (PR) Curves - Multi-Model Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pr_vs_path = os.path.join(settings.OUTPUT_DIR, "combined_pr_curves_vs.png")
    plt.savefig(pr_vs_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pr_vs_path}", flush=True)

    # 3. Multi-Metric Grouped Bar Chart
    metrics_to_plot = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc", "mcc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC", "MCC"]

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    x = np.arange(len(metric_labels))
    width = 0.20
    models_list = list(metrics.keys())

    for idx, name in enumerate(models_list):
        scores = [metrics[name].get(m, 0.0) for m in metrics_to_plot]
        color = model_colors.get(name, "#333333")
        offset = x + (idx - len(models_list) / 2.0 + 0.5) * width
        rects = ax.bar(offset, scores, width, label=model_display_names.get(name, name), color=color, edgecolor="black", linewidth=0.5, alpha=0.9)

        # Label values on bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Comprehensive Evaluation Metrics Across Deep Learning Models", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylim([0.0, 1.15])
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    barchart_path = os.path.join(settings.OUTPUT_DIR, "model_metrics_comparison_barchart.png")
    plt.savefig(barchart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {barchart_path}", flush=True)

    # 4. Latency vs. Accuracy Real-Time Efficiency Scatter Plot
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    for name, m in metrics.items():
        color = model_colors.get(name, "#333333")
        acc_pct = m["accuracy"] * 100.0
        lat_ms = m["latency_ms"]
        fps = m["fps"]

        ax.scatter(lat_ms, acc_pct, s=350, color=color, alpha=0.85, edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(f"{model_display_names.get(name, name)}\n({acc_pct:.2f}%, {lat_ms:.2f} ms)",
                    xy=(lat_ms, acc_pct), xytext=(12, -8), textcoords="offset points",
                    fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5))

    ax.set_xlabel("Inference Latency per Image (ms) [Lower is Better]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%) [Higher is Better]", fontsize=12, fontweight="bold")
    ax.set_title("Real-Time Deployment Trade-off: Latency vs. Accuracy", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    scatter_path = os.path.join(settings.OUTPUT_DIR, "latency_vs_accuracy_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {scatter_path}", flush=True)

    # 5. Combined Confusion Matrix Grid (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), dpi=300)
    axes = axes.flatten()

    for idx, name in enumerate(models_list[:4]):
        ax = axes[idx]
        cm_dict = metrics[name]["confusion_matrix"]
        cm_matrix = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])

        sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{model_display_names.get(name, name)}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Actual Label", fontsize=10)

    plt.suptitle("Confusion Matrix Grid Panel Comparison", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    grid_path = os.path.join(settings.OUTPUT_DIR, "confusion_matrix_grid.png")
    plt.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {grid_path}", flush=True)

    # 6. Combined Training & Validation Loss/Accuracy Curves across Epochs
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    for name in models_list:
        hist_file = os.path.join(settings.OUTPUT_DIR, f"{name}_history.json")
        if not os.path.exists(hist_file):
            continue
        with open(hist_file, "r") as f:
            hist = json.load(f)

        epochs_range = range(1, len(hist.get("train_loss", [])) + 1)
        color = model_colors.get(name, "#333333")
        label = model_display_names.get(name, name)

        if "train_loss" in hist and hist["train_loss"]:
            ax1.plot(epochs_range, hist["train_loss"], lw=2, color=color, label=label)
        if "val_loss" in hist and hist["val_loss"]:
            ax2.plot(epochs_range, hist["val_loss"], lw=2, color=color, label=label)
        if "train_acc" in hist and hist["train_acc"]:
            ax3.plot(epochs_range, hist["train_acc"], lw=2, color=color, label=label)
        if "val_acc" in hist and hist["val_acc"]:
            ax4.plot(epochs_range, hist["val_acc"], lw=2, color=color, label=label)

    ax1.set_title("Training Loss Convergence", fontweight="bold")
    ax1.set_xlabel("Epochs"); ax1.set_ylabel("Loss"); ax1.grid(True, linestyle="--", alpha=0.5); ax1.legend()

    ax2.set_title("Validation Loss Convergence", fontweight="bold")
    ax2.set_xlabel("Epochs"); ax2.set_ylabel("Loss"); ax2.grid(True, linestyle="--", alpha=0.5); ax2.legend()

    ax3.set_title("Training Accuracy (%)", fontweight="bold")
    ax3.set_xlabel("Epochs"); ax3.set_ylabel("Accuracy (%)"); ax3.grid(True, linestyle="--", alpha=0.5); ax3.legend()

    ax4.set_title("Validation Accuracy (%)", fontweight="bold")
    ax4.set_xlabel("Epochs"); ax4.set_ylabel("Accuracy (%)"); ax4.grid(True, linestyle="--", alpha=0.5); ax4.legend()

    plt.suptitle("Multi-Model Training & Validation Convergence Comparison", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    train_vs_path = os.path.join(settings.OUTPUT_DIR, "combined_training_curves_vs.png")
    plt.savefig(train_vs_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {train_vs_path}", flush=True)

if __name__ == "__main__":
    metrics, curves_data = compute_all_metrics()
    generate_all_plots(metrics, curves_data)
