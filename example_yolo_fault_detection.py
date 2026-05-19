import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path("models/faultnet_yolo/best.pt")
DEFAULT_OUTPUT = Path("outputs/faultnet_yolo_example.png")


def load_rgb_image(path):
    array = np.asarray(Image.open(path))
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[-1] in (3, 4) and array.dtype == np.uint8:
        return array[..., :3].copy()
    if array.ndim == 3:
        array = array[..., 0]
    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError(f"Image has no finite pixel values: {path}")
    array = np.where(finite, array, np.nanmedian(array[finite]))
    low, high = np.percentile(array[finite], [1, 99])
    if abs(float(high - low)) < 1e-6:
        low, high = float(array[finite].min()), float(array[finite].max())
    array = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
    array = (array * 255).astype(np.uint8)
    return np.repeat(array[..., None], 3, axis=-1)


def run_yolo_faultnet(model, image, conf=0.25):
    results = model.predict(source=image, conf=conf, verbose=False)
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist() if getattr(boxes, "cls", None) is not None else [None] * len(xyxy)
        names = getattr(result, "names", {}) or {}
        masks = getattr(result, "masks", None)
        mask_data = masks.data.detach().cpu() if masks is not None and getattr(masks, "data", None) is not None else None
        polygons = masks.xy if masks is not None and getattr(masks, "xy", None) is not None else []

        for index, (box, score, class_id) in enumerate(zip(xyxy, confs, classes)):
            label = names.get(int(class_id), "fault") if class_id is not None else "fault"
            detection = {
                "label": str(label),
                "confidence": round(float(score), 4),
                "box_xyxy": [round(float(value), 2) for value in box],
            }
            if mask_data is not None and index < len(mask_data):
                mask = mask_data[index].float()
                detection["mask_coverage"] = round(float(mask.mean().item()), 6)
            if index < len(polygons) and len(polygons[index]) > 0:
                polygon = polygons[index]
                step = max(len(polygon) // 16, 1)
                detection["polygon_xy_sample"] = [
                    [round(float(x), 2), round(float(y), 2)]
                    for x, y in polygon[::step][:16]
                ]
            detections.append(detection)
    return detections


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

    model = YOLO(args.weights)
    image = load_rgb_image(args.image)
    if args.save_input_preview:
        preview_path = Path(args.save_input_preview)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(preview_path)
    detections = run_yolo_faultnet(model, image, conf=args.conf)
    overlay = draw_detection_overlay(image, detections, args.conf)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)

    print(json.dumps(detections, indent=2))
    print(f"detections: {len(detections)} at conf={args.conf}")
    print(f"saved overlay: {output_path}")


if __name__ == "__main__":
    main()
