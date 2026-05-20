from pathlib import Path

import numpy as np

from seismic_k2.dataset_generator.image_utils import load_image


def load_faultnet(weights_path):
    if weights_path is None:
        return None
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "FaultNet YOLO inference needs `ultralytics`. Install it or omit --faultnet-weights."
        ) from error

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"FaultNet weights not found: {weights_path}")
    return YOLO(weights_path.as_posix())


def run_faultnet(model, image_path, conf=0.25):
    if model is None:
        return []
    image = np.asarray(load_image(image_path))
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
