# Model Comparison & Quantitative Benchmark Report

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC | Log Loss | Latency (ms) | Throughput (FPS) | Parameters | Checkpoint Size |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FASTCNN (Custom CNN)** | **84.78%** | 79.58% | **99.36%** | 64.42% | 88.38% | 0.9883 | 0.9910 | 0.7063 | 0.4332 | **0.17 ms** | **5,968 FPS** | **0.25M** | **0.96 MB** |
| **MobileNetV2** | **92.31%** | **98.99%** | 87.70% | **98.75%** | 93.00% | **0.9937** | **0.9952** | **0.8531** | 0.1823 | 0.37 ms | 2,727 FPS | 2.23M | 8.72 MB |
| **Recurrent CNN (RCNN)** | 91.97% | 91.40% | 95.17% | 87.50% | **93.24%** | 0.9801 | 0.9884 | 0.8345 | **0.1815** | **0.17 ms** | **6,054 FPS** | 4.77M | 18.21 MB |
| **EfficientNet-B0** | 51.31% | 63.06% | 39.67% | 67.57% | 48.70% | 0.5024 | 0.6641 | 0.0741 | 7.5176 | 0.48 ms | 2,090 FPS | 4.01M | 46.34 MB |

### Key Takeaways:
- **Safety Sentinel (FASTCNN):** Highest sensitivity/recall (**99.36%**, only 30 missed cracks out of 4,699 positive test samples) with ultra-low latency (**0.17 ms**, 5,968 FPS) and 0.96 MB size.
- **Accuracy Champion (MobileNetV2):** Highest test accuracy (**92.31%**) and precision (**98.99%**) with only 42 false alarms across 8,066 test tiles.
- **Recurrent Spatial Modeler (RCNN):** Excellent balance of high accuracy (**91.97%**), high recall (**95.17%**), and peak F1-score (**93.24%**).

