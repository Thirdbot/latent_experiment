import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from seismic_k2.dataset_generator.image_utils import load_image
from seismic_k2.fault.detection import load_faultnet, run_faultnet


DEFAULT_WEIGHTS = Path("models/faultnet_yolo/best.pt")
DEFAULT_OUTPUT = Path("outputs/faultnet_yolo_example.png")


def draw_status(overlay, text):
    cv2.rectangle(overlay, (0, 0), (min(overlay.shape[1], 520), 42), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 220, 255),
        2,
        cv2.LINE_AA,
    )


def draw_detection_overlay(image, detections, conf):
    overlay = image.copy()
    color = (255, 40, 40)
    draw_status(overlay, f"detections={len(detections)} conf={conf:.3f}")

    if not detections:
        cv2.putText(
            overlay,
            "no fault detections",
            (12, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 40, 40),
            2,
            cv2.LINE_AA,
        )
        return overlay

    for detection in detections:
        box = detection.get("box_xyxy")
        if box:
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = detection.get("label", "fault")
            score = detection.get("confidence", 0.0)
            cv2.putText(
                overlay,
                f"{label} {score:.2f}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        polygon = detection.get("polygon_xy_sample") or []
        if len(polygon) >= 2:
            points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [points], isClosed=True, color=(40, 220, 255), thickness=2)

    return overlay


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
