import os
import argparse
import base64
import time
from PIL import Image
import torch

from backend.services import InferenceService

def parse_args():
    parser = argparse.ArgumentParser(description="FASTCNN Inference Script for Custom Images")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image file")
    parser.add_argument("--dir", type=str, default=None, help="Path to a directory containing images")
    parser.add_argument("--model", type=str, default="custom_cnn", help="Model name: custom_cnn, mobilenet_v2, efficientnet, onnx_quantized")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Directory to save visual output heatmaps and results")
    parser.add_argument("--no-heatmap", action="store_true", help="Disable Grad-CAM heatmap overlay generation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.image and not args.dir:
        print("Please provide either --image <path_to_image> or --dir <path_to_directory>.")
        print("Example usage:")
        print("  python predict.py --image path/to/concrete.jpg")
        print("  python predict.py --dir path/to/my_images/ --model custom_cnn")
        return

    # Collect images
    image_paths = []
    if args.image:
        if os.path.exists(args.image):
            image_paths.append(args.image)
        else:
            print(f"Error: Image path '{args.image}' does not exist.")
            return
    elif args.dir:
        if os.path.exists(args.dir):
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
            for root, _, files in os.walk(args.dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        image_paths.append(os.path.join(root, f))
            if not image_paths:
                print(f"No valid image files found in directory '{args.dir}'.")
                return
        else:
            print(f"Error: Directory path '{args.dir}' does not exist.")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading inference service with model: '{args.model}'...")
    service = InferenceService()
    service.load_model(args.model)

    generate_heatmap = not args.no_heatmap
    print(f"\nProcessing {len(image_paths)} image(s)...\n" + "=" * 60)

    for idx, path in enumerate(image_paths, 1):
        filename = os.path.basename(path)
        with open(path, "rb") as f:
            img_bytes = f.read()

        result = service.predict(img_bytes, filename=filename, generate_heatmap=generate_heatmap)
        
        print(f"[{idx}/{len(image_paths)}] File: {filename}")
        print(f"  Prediction:      {result['prediction']}")
        print(f"  Confidence:      {result['confidence'] * 100:.2f}%")
        print(f"  Probability:     {result['probability']:.4f}")
        print(f"  Inference Time:  {result['inference_time_ms']:.2f} ms")

        # Save heatmap if generated
        if result.get("heatmap_image"):
            b64_data = result["heatmap_image"]
            img_data = base64.b64decode(b64_data)
            save_name = f"gradcam_{os.path.splitext(filename)[0]}.png"
            save_path = os.path.join(args.output_dir, save_name)
            with open(save_path, "wb") as f_out:
                f_out.write(img_data)
            print(f"  Saved Heatmap:   {save_path}")
        print("-" * 60)

    print(f"\nFinished! Results saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
