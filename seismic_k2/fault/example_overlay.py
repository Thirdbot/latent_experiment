import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from seismic_k2.dataset_generator.image_utils import load_image
from seismic_k2.fault.detection import load_faultnet, run_faultnet
from seismic_k2.fault.overlay import draw_detection_overlay


DEFAULT_WEIGHTS = Path("models/faultnet_yolo/best.pt")
DEFAULT_OUTPUT = Path("outputs/faultnet_yolo_example.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a seismic image.")
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS.as_posix(),
        help="Path to trained YOLO segmentation weights.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT.as_posix(),
        help="Path to save the overlay image.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--save-input-preview",
        default=None,
        help="Optional path to save the normalized RGB image before drawing detections.",
    )
    args = parser.parse_args()

    model = load_faultnet(args.weights)
    image = np.asarray(load_image(args.image))
    if args.save_input_preview:
        preview_path = Path(args.save_input_preview)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(preview_path)
    detections = run_faultnet(model, args.image, conf=args.conf)
    overlay = draw_detection_overlay(image, detections, args.conf)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)

    print(json.dumps(detections, indent=2))
    print(f"detections: {len(detections)} at conf={args.conf}")
    print(f"saved overlay: {output_path}")


if __name__ == "__main__":
    main()
