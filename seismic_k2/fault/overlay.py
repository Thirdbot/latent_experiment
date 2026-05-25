from __future__ import annotations

import cv2
import numpy as np


def draw_status(overlay: np.ndarray, text: str) -> None:
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


def draw_detection_overlay(image: np.ndarray, detections: list[dict], conf: float) -> np.ndarray:
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
