# Concrete Crack Detection Debugging & Root-Cause Analysis Report

This report documents the detailed investigation, findings, and resolution of the bug where the Custom Fast CNN model was returning a static probability of approximately **49.3%** for all input images regardless of their visual contents.

---

## 1. Executive Summary

- **Problem:** Every image ran through the backend inference server yielded a confidence/probability of **~49.3%**.
- **Root Cause:** The system diagnostics and verification script (`verify_pipeline.py`) had a destructive testing logic in its ONNX verification phase. It instantiated an untrained, randomly initialized dummy model, saved it to the production checkpoint path `custom_cnn_best.pth` (overwriting the actual trained weights), exported these dummy weights to `custom_cnn.onnx` and `custom_cnn_quantized.onnx`, and then deleted `custom_cnn_best.pth` entirely. 
- **Mechanism of Bug:** Because the best checkpoint was deleted, the backend server fell back to initializing a raw, untrained model architecture. An untrained network with randomly initialized weights outputs logits close to $0.0$. In this network, the raw logit output was approximately $-0.028$, and $\text{sigmoid}(-0.028) \approx 49.3\%$.
- **Severity:** Critical (caused complete model degradation during inference).
- **Resolution:** 
  1. Restored the trained weights from the epoch-14 checkpoint (`custom_cnn_latest.pth`) back to `custom_cnn_best.pth`.
  2. Re-exported the true trained weights to the ONNX and quantized ONNX formats.
  3. Modified `verify_pipeline.py` to check for the presence of the trained checkpoint and safely use it or a separate temporary path (`custom_cnn_temp.pth`) without ever deleting or overwriting production weights.
  4. Successfully verified that the backend loads the trained weights and outputs correct predictions (near $100\%$ for cracks and $<5\%$ for non-cracks).

---

## 2. Comprehensive Pipeline Inspection

### 1. Dataset Check
- **Files Location:** Checked `kaggle_dataset/Positive` and `kaggle_dataset/Negative`.
- **Class Distribution:** Exactly 20,000 Positive (Crack) and 20,000 Negative (No Crack) images, totaling 40,000 images (perfectly balanced 50.0% split).
- **Corruption Check:** 0 corrupted files detected.
- **Duplicates Check:** 1,598 files (4.0% of the dataset) identified as duplicate pixel arrays (typical in structural crack datasets).
- **Normalization:** Train splits use ImageNet standardization (mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]`) which is standard for transfer learning.

### 2. Dataloader Check
- **Batch Shapes:** Generates expected `[32, 3, 224, 224]` tensors.
- **Augmentation:** Employs appropriate spatial augmentations (horizontal/vertical flips, small rotations/affine shifts) and high-frequency noise injection (`AddGaussianNoise(std=0.03)`).
- **Correctness:** Labels are correctly mapped: `1.0` for Positive (Crack) and `0.0` for Negative (No Crack).

### 3. CNN Architecture Check
- **Model:** `CustomFastCNN` utilizes standard Conv layers followed by `DepthwiseSeparableConv` blocks to minimize parameters while maintaining spatial extraction capabilities.
- **Activation Functions:** ReLU is used throughout the feature extractor.
- **Classifier Head:** A linear layer projects the 512-channel pooled output to a single raw logit (`num_classes=1`).
- **Binary Classification Setup:** The loss function used during training is `nn.BCEWithLogitsLoss()`. It expects raw logits from the model, and then applies a stable sigmoid during loss computation, which is correct and numerically stable.

### 4. Training Process Logs Check
- **Source:** `outputs/custom_cnn_history.json`
- **Analysis:**
  - The model finished training all 15 epochs.
  - Final Train Accuracy: **98.26%** (Loss: `0.0523`)
  - Final Validation Accuracy: **99.65%** (Loss: `0.0206`)
  - The loss curves showed smooth convergence with no signs of underfitting, exploding gradients, or vanishing gradients.

![Training History Plot](file:///C:/Users/Mayur%20Jadhav/.gemini/antigravity-ide/brain/7278ce09-8429-4bcc-916f-db0a85b34693/custom_cnn_training_history.png)

### 5. Checkpoint & Saved Model Check
- **Trained Checkpoint:** `custom_cnn_latest.pth` was found in the checkpoints folder. It was verified to contain fully trained weights (mean absolute parameter difference of $\sim 0.12$ compared to randomly initialized weights).
- **Missing Best Weights:** `custom_cnn_best.pth` was missing from the checkpoints folder.
- **ONNX Models:** `custom_cnn.onnx` and `custom_cnn_quantized.onnx` were present but their size and outputs indicated they contained randomly initialized weights.

### 6. Prediction Pipeline Check
- **Preprocessing:** Preprocessing in `backend/services.py` matches validation preprocessing in `training/dataset.py` exactly (resize to $224\times 224$, conversion to tensor, and ImageNet normalization).
- **Bug Source:** The backend attempts to load weights from `custom_cnn_best.pth`. Since this file was missing, the backend fell back to using the untrained architecture.

---

## 3. The Root Cause: Destructive Diagnostics

The bug was introduced in `verify_pipeline.py` inside the `verify_onnx_export_pipeline` function. Here is the original destructive code:

```python
def verify_onnx_export_pipeline():
    """Export custom cnn to ONNX, quantize it and check if it runs using ONNX Runtime."""
    logger.info("\n=== STEP 5: ONNX & Edge Optimization Verification ===")
    
    # Create a temporary dummy weights file since export requires a checkpoint
    dummy_model = get_model("custom_cnn", pretrained=False, num_classes=1)
    temp_path = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn_best.pth")
    torch.save(dummy_model.state_dict(), temp_path) # <--- OVERWRITING TRAINED WEIGHTS WITH RANDOM WEIGHTS!
    
    try:
        # Export
        onnx_path = export_to_onnx("custom_cnn", model_path=temp_path) # <--- EXPORTING RANDOM WEIGHTS!
        ...
        # Clean up temporary weights
        if os.path.exists(temp_path):
            os.remove(temp_path) # <--- DELETING THE BEST WEIGHTS CHECKPOINT ENTIRELY!
```

### Impact:
1. **Destruction of PyTorch checkpoint:** The best model weights at `custom_cnn_best.pth` were overwritten and then deleted.
2. **Corruption of ONNX models:** The production ONNX models (`custom_cnn.onnx` and `custom_cnn_quantized.onnx`) were exported from the untrained dummy model, rendering ONNX inference completely useless.
3. **API Static Output:** The backend server started up, warned that `custom_cnn_best.pth` was not found, initialized a raw model architecture, and ran inference on random weights, returning the static **49.3%** prediction.

---

## 4. Correction & System Restoration

We restored the checkpoints and fixed the verification script to prevent future regression.

### 1. Checkpoint Restoration Code
We executed a recovery script to copy the trained state dict from the epoch-14 checkpoint (`custom_cnn_latest.pth`) back to `custom_cnn_best.pth` and re-export the true trained weights to ONNX:

```python
latest_path = os.path.join(checkpoint_dir, "custom_cnn_latest.pth")
best_path = os.path.join(checkpoint_dir, "custom_cnn_best.pth")

# Load and restore
checkpoint = torch.load(latest_path, map_location="cpu")
torch.save(checkpoint["model_state_dict"], best_path)

# Re-export ONNX files from the restored best checkpoint
export_to_onnx("custom_cnn", model_path=best_path)
quantize_onnx_model("checkpoints/custom_cnn.onnx")
```

### 2. Safely Modifying the Verification Script
We replaced the destructive function in `verify_pipeline.py` with a non-destructive logic:

```diff
-    # Create a temporary dummy weights file since export requires a checkpoint
-    dummy_model = get_model("custom_cnn", pretrained=False, num_classes=1)
-    temp_path = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn_best.pth")
-    torch.save(dummy_model.state_dict(), temp_path)
-    
-    try:
-        # Export
-        onnx_path = export_to_onnx("custom_cnn", model_path=temp_path)
-        logger.info(f"ONNX export verified: file exists = {os.path.exists(onnx_path)}")
-        
-        # Quantize
-        quant_path = quantize_onnx_model(onnx_path)
-        logger.info(f"ONNX quantization verified: file exists = {os.path.exists(quant_path)}")
-        
-        # Clean up temporary weights
-        if os.path.exists(temp_path):
-            os.remove(temp_path)
+    best_path = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn_best.pth")
+    
+    if os.path.exists(best_path):
+        logger.info("Trained model checkpoint found. Using it for ONNX verification...")
+        model_path = best_path
+        cleanup = False
+    else:
+        logger.info("Trained model checkpoint not found. Creating a temporary dummy model for verification...")
+        dummy_model = get_model("custom_cnn", pretrained=False, num_classes=1)
+        model_path = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn_temp.pth")
+        torch.save(dummy_model.state_dict(), model_path)
+        cleanup = True
+    
+    try:
+        # Export
+        onnx_path = export_to_onnx("custom_cnn", model_path=model_path)
+        logger.info(f"ONNX export verified: file exists = {os.path.exists(onnx_path)}")
+        
+        # Quantize
+        quant_path = quantize_onnx_model(onnx_path)
+        logger.info(f"ONNX quantization verified: file exists = {os.path.exists(quant_path)}")
+        
+        # Clean up temporary weights if we created them
+        if cleanup and os.path.exists(model_path):
+            os.remove(model_path)
```

---

## 5. Verification Metrics and Inferences

### 1. Inferences on 20 Test Images
We evaluated the restored model on 10 positive (cracked) and 10 negative (non-cracked) images randomly sampled from the test split:

| Filename | Ground Truth | Raw Logit | Probability | Prediction |
| :--- | :--- | :--- | :--- | :--- |
| `03160.jpg` | **Crack** | `+15.9106` | **100.0%** | **Crack** |
| `00185.jpg` | **Crack** | `+10.7064` | **100.0%** | **Crack** |
| `16587_1.jpg` | **Crack** | `+11.3029` | **100.0%** | **Crack** |
| `05321.jpg` | **Crack** | `+4.6028` | **99.0%** | **Crack** |
| `08143.jpg` | **Crack** | `+10.0558` | **100.0%** | **Crack** |
| `02705.jpg` | **Crack** | `+13.9911` | **100.0%** | **Crack** |
| `13604_1.jpg` | **Crack** | `+9.3591` | **100.0%** | **Crack** |
| `18196_1.jpg` | **Crack** | `+8.2946` | **100.0%** | **Crack** |
| `09226.jpg` | **Crack** | `+12.1209` | **100.0%** | **Crack** |
| `14933_1.jpg` | **Crack** | `+12.6667` | **100.0%** | **Crack** |
| `02525.jpg` | **No Crack** | `-3.1259` | **4.2%** | **No Crack** |
| `02051.jpg` | **No Crack** | `-4.1542` | **1.5%** | **No Crack** |
| `05883.jpg` | **No Crack** | `-4.6293` | **1.0%** | **No Crack** |
| `09551.jpg` | **No Crack** | `-3.5795` | **2.7%** | **No Crack** |
| `01921.jpg` | **No Crack** | `-5.7657` | **0.3%** | **No Crack** |
| `08682.jpg` | **No Crack** | `-3.9493` | **1.9%** | **No Crack** |
| `19986.jpg` | **No Crack** | `-3.3820` | **3.3%** | **No Crack** |
| `00613.jpg` | **No Crack** | `-8.1008` | **0.0%** | **No Crack** |
| `12381.jpg` | **No Crack** | `-4.3823` | **1.2%** | **No Crack** |
| `16544.jpg` | **No Crack** | `-5.3743` | **0.5%** | **No Crack** |

> [!NOTE]
> The model achieves **100% classification accuracy** on this test set, displaying highly confident logits that perfectly distinguish between crack and non-crack classes.

---

### 2. Diagnostic Tests
To inspect extreme data bounds and edge cases, we executed the 5 requested diagnostic tests:

- **Test A: One positive image repeated 10 times (`03160.jpg`)**
  - **Output:** Every sample yielded a logit of `+15.9106` and probability of `100.0%` (perfect reproducibility).
- **Test B: One negative image repeated 10 times (`02525.jpg`)**
  - **Output:** Every sample yielded a logit of `-3.1259` and probability of `4.2%` (perfect reproducibility).
- **Test C: Random noise image repeated 5 times**
  - **Output:** Every sample yielded a logit of `+40.3066` and probability of `100.0%`.
  - *Engineering Insight:* High-frequency random noise triggers edge detection filters in early convolution layers. Because there are no smoothing regularizations or low-pass constraints, it produces strong activations that mimic structural crack edges.
- **Test D: Completely black image repeated 5 times**
  - **Output:** Every sample yielded a logit of `-0.1042` and probability of `47.4%`.
  - *Engineering Insight:* Standardized black pixels ($[0,0,0]$) map to large negative inputs due to subtracting the ImageNet mean. The lack of standard edges pulls the activation down near neutral/slightly negative logits.
- **Test E: Completely white image repeated 5 times**
  - **Output:** Every sample yielded a logit of `-5.7791` and probability of `0.3%`.
  - *Engineering Insight:* Flat white images translate to uniform activations that strongly activate inhibitory pathways or trigger no edge filters, pushing the probability down to near zero.

---

## 6. Feature Visualization & Localization

To confirm that the Custom Fast CNN model is indeed learning meaningful structural crack features rather than caching background noise, we analyzed the activation maps and Grad-CAM overlays:

### Grad-CAM Overlay
The Grad-CAM overlay shows that the gradients focus tightly along the high-contrast linear fissure patterns on the concrete surfaces.

![Grad-CAM Overlay](file:///C:/Users/Mayur%20Jadhav/.gemini/antigravity-ide/brain/7278ce09-8429-4bcc-916f-db0a85b34693/feature_vis_gradcam.png)

### Intermediate Feature Maps
- **Conv1 (Low-level):** Highlights edge-like features, capturing pixel-level transitions.
- **Block 1 & 2 (Mid-level):** Activates along the specific contours of the crack, isolating the structural crack line from ambient surface texture.
- **Block 3 & 4 (High-level / Semantic):** Aggregates these details into a semantic representation of a crack, resulting in localized blobs near the crack's path.

![Feature Maps Summary](file:///C:/Users/Mayur%20Jadhav/.gemini/antigravity-ide/brain/7278ce09-8429-4bcc-916f-db0a85b34693/feature_vis_summary.png)

---

## 7. Operational Recommendations

1. **Write-Protection on Checkpoints:** Implement read-only permissions on checkpoints in CI/CD or add unit tests that check if a checkpoint has changed size or hash during testing.
2. **Inference Fallback Guard:** Modify `backend/services.py` to raise a hard `FileNotFoundError` during backend startup if the best model checkpoint is missing, rather than silently falling back to untrained weights. A failing service is always easier to debug than a degraded service.
3. **Quantization Calibration:** To improve the INT8 quantized model accuracy, run post-training calibration with representative validation images using the `onnxruntime.quantization.calibrate` tool rather than dynamic quantization.
