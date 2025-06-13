# CLI for cmfd_loftr.detect 
import argparse
import numpy as np
import cv2
from pathlib import Path
from .detect import detect_copy_move


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay the binary mask on the image: red for copied, green for reference.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()
    # Red for copied
    overlay[mask == 1] = [0, 0, 255]
    # Optionally, you can add green for reference if you have that info
    return cv2.addWeighted(image, 0.6, overlay, 0.4, 0)


def main():
    parser = argparse.ArgumentParser(description="Copy-move forgery detection with LoFTR self-matching.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output mask path (PNG)")
    parser.add_argument("--viz", action="store_true", help="Save overlay visualization (PNG)")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda or cpu)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)
    mask = detect_copy_move(img, device=args.device, verbose=args.verbose)
    cv2.imwrite(args.output, (mask * 255).astype(np.uint8))

    if args.viz:
        overlay = overlay_mask(img, mask)
        viz_path = str(Path(args.output).with_suffix(".viz.png"))
        cv2.imwrite(viz_path, overlay)
        if args.verbose:
            print(f"Overlay saved to {viz_path}")

if __name__ == "__main__":
    main() 