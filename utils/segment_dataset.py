import os
import sys
import glob
from PIL import Image
import torch
from torchvision import transforms

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings, BASE_DIR
from models.factory import get_model

def segment_raw_images(
    source_dir: str = str(BASE_DIR / "kaggle_dataset"),
    output_pos_dir: str = None,
    output_neg_dir: str = None,
    patch_size: int = 224,
    stride: int = 224,
    confidence_threshold: float = 0.50
):
    """
    Cropping high-res PNG/JPG images in source_dir into patch_size x patch_size tiles.
    Uses the trained FASTCNN model to classify each tile into Positive (crack) and Negative (no crack).
    """
    if output_pos_dir is None:
        output_pos_dir = os.path.join(source_dir, "Positive")
    if output_neg_dir is None:
        output_neg_dir = os.path.join(source_dir, "Negative")

    os.makedirs(output_pos_dir, exist_ok=True)
    os.makedirs(output_neg_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(source_dir, "*.png")) + glob.glob(os.path.join(source_dir, "*.jpg")) + glob.glob(os.path.join(source_dir, "*.jpeg")))
    print(f"Found {len(image_files)} raw images in '{source_dir}' to segment.")

    if not image_files:
        print("No raw image files found for segmentation.")
        return

    # Load model for segmentation inference
    device = torch.device(settings.DEVICE)
    model_name = "custom_cnn"
    model_path = os.path.join(settings.CHECKPOINT_DIR, f"{model_name}_best.pth")

    model_available = False
    if os.path.exists(model_path):
        try:
            model = get_model(model_name, pretrained=False, num_classes=1, freeze_backbone=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            model_available = True
            print(f"Loaded existing trained checkpoint '{model_path}' for auto-annotation.")
        except Exception as e:
            print(f"Could not load model checkpoint: {e}. Falling back to variance/contrast thresholding.")
    else:
        print(f"No checkpoint found at '{model_path}'. Falling back to feature-based segmentation.")

    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pos_count = 0
    neg_count = 0
    patch_idx = 1
    total_imgs = len(image_files)

    for i, img_path in enumerate(image_files, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                w, h = img.size

                crops = []
                coords = []
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride):
                        box = (x, y, x + patch_size, y + patch_size)
                        crops.append(img.crop(box))
                        coords.append((x, y))

                if not crops:
                    continue

                if model_available:
                    batch_tensors = torch.stack([transform(c) for c in crops]).to(device)
                    with torch.no_grad():
                        logits = model(batch_tensors)
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                else:
                    probs = []
                    for c in crops:
                        gray = c.convert("L")
                        extrema = gray.getextrema()
                        probs.append(1.0 if (extrema[1] - extrema[0]) > 80 else 0.0)

                for (x, y), crop, prob in zip(coords, crops, probs):
                    is_positive = (prob >= confidence_threshold)
                    save_filename = f"patch_{patch_idx:06d}_{base_name}_x{x}_y{y}.png"
                    if is_positive:
                        save_path = os.path.join(output_pos_dir, save_filename)
                        crop.save(save_path)
                        pos_count += 1
                    else:
                        save_path = os.path.join(output_neg_dir, save_filename)
                        crop.save(save_path)
                        neg_count += 1
                    patch_idx += 1

            if i % 10 == 0 or i == total_imgs:
                print(f"[{i}/{total_imgs}] Processed {img_path} | Cumulative Patches: Pos={pos_count}, Neg={neg_count}", flush=True)

        except Exception as e:
            print(f"Error processing image {img_path}: {e}", flush=True)

    print(f"\nSegmentation complete!", flush=True)
    print(f"Extracted Patches -> Positive (Crack): {pos_count} | Negative (No Crack): {neg_count}", flush=True)

if __name__ == "__main__":
    segment_raw_images()
