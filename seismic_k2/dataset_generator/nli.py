from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ID_TO_VERIFIER_LABEL = {
    0: "SUPPORT",
    1: "UNCERTAIN",
    2: "CONTRADICT",
    3: "OVERCLAIM",
}
REMOVE_LABELS = {"CONTRADICT", "OVERCLAIM"}

FAULT_TERMS = re.compile(r"\b(fault|faulted|faulting|fracture|throw|offset|discontinuity|disrupted)\b", re.I)
ABSENCE_TERMS = re.compile(r"\b(no|none|absent|without|not detected|not visible|does not show|do not show)\b", re.I)
CERTAINTY_TERMS = re.compile(r"\b(definitely|certainly|proves|clearly indicates|high-confidence|unambiguous)\b", re.I)
UNCERTAINTY_TERMS = re.compile(r"\b(possible|possibly|likely|may|might|suggests|ambiguous|uncertain)\b", re.I)


def verify_generated_dataset(
    input_path: Path,
    output_path: Path,
    filtered_output_path: Path | None = None,
    model_path: Path | None = None,
    max_length: int = 256,
) -> dict[str, Any]:
    verifier = load_model_verifier(model_path, max_length) if model_path else verify_pair_with_rules
    verified_rows = []
    kept_rows = []
    stats = Counter()
    missing = 0

    for row in read_jsonl(input_path):
        nli = build_nli(row)
        premise = nli["premise"]
        hypothesis = nli["hypothesis"]
        if not premise or not hypothesis:
            missing += 1
            label, confidence, scores = "UNCERTAIN", 0.0, {}
        else:
            label, confidence, scores = verifier(premise, hypothesis)

        row["nli"] = nli
        row["verification"] = {
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "remove": label in REMOVE_LABELS,
        }
        stats[label] += 1
        verified_rows.append(row)
        if label not in REMOVE_LABELS:
            kept_rows.append(row)

    write_jsonl(output_path, verified_rows)
    if filtered_output_path is not None:
        write_jsonl(filtered_output_path, kept_rows)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "filtered_output": str(filtered_output_path) if filtered_output_path else None,
        "total": len(verified_rows),
        "kept": len(kept_rows),
        "removed": len(verified_rows) - len(kept_rows),
        "missing_premise_or_hypothesis": missing,
        "labels": dict(stats),
    }


def build_nli(row: dict[str, Any]) -> dict[str, Any]:
    detections = row.get("faultnet_detections") or []
    answer = row.get("merged_answer") or row.get("generator_answer") or row.get("answer") or ""
    premise = make_fault_premise(detections)
    hypothesis = normalize_hypothesis(answer)
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "claim_type": "fault_overlay_claim",
        "variables": fault_evidence_variables(detections),
    }


def make_fault_premise(detections: list[dict[str, Any]]) -> str:
    if not detections:
        return (
            "Auxiliary FaultNet produced no fault detections for this image. "
            "This absence is weak evidence only and does not prove faults are absent."
        )

    scores = [float(detection.get("confidence", 0.0)) for detection in detections]
    max_score = max(scores) if scores else 0.0
    boxes = sum(1 for detection in detections if detection.get("box_xyxy"))
    polygons = sum(1 for detection in detections if detection.get("polygon_xy_sample"))
    return (
        f"Auxiliary FaultNet produced {len(detections)} fault detection(s), "
        f"{boxes} boxed region(s), and {polygons} polygon overlay sample(s). "
        f"The maximum detection confidence is {max_score:.3f}."
    )


def normalize_hypothesis(answer: str) -> str:
    text = " ".join(str(answer).split())
    if len(text) > 600:
        text = text[:597].rsplit(" ", 1)[0] + "..."
    return text


def fault_evidence_variables(detections: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(detection.get("confidence", 0.0)) for detection in detections]
    return {
        "fault_present": bool(detections),
        "detection_count": len(detections),
        "max_confidence": max(scores) if scores else 0.0,
        "has_box_overlay": any(detection.get("box_xyxy") for detection in detections),
        "has_polygon_overlay": any(detection.get("polygon_xy_sample") for detection in detections),
    }


def verify_pair_with_rules(premise: str, hypothesis: str) -> tuple[str, float, dict[str, float]]:
    premise_fault = "produced no fault detections" not in premise.lower()
    premise_weak_absence = "weak evidence only" in premise.lower()
    claims_fault = FAULT_TERMS.search(hypothesis) is not None
    claims_absence = ABSENCE_TERMS.search(hypothesis) is not None and claims_fault
    overconfident = CERTAINTY_TERMS.search(hypothesis) is not None
    hedged = UNCERTAINTY_TERMS.search(hypothesis) is not None

    if not claims_fault:
        return "UNCERTAIN", 0.55, {"UNCERTAIN": 0.55, "SUPPORT": 0.25, "CONTRADICT": 0.15, "OVERCLAIM": 0.05}
    if premise_fault and claims_absence:
        return "CONTRADICT", 0.80, {"CONTRADICT": 0.80, "UNCERTAIN": 0.10, "SUPPORT": 0.05, "OVERCLAIM": 0.05}
    if premise_fault and overconfident:
        return "OVERCLAIM", 0.70, {"OVERCLAIM": 0.70, "SUPPORT": 0.15, "UNCERTAIN": 0.10, "CONTRADICT": 0.05}
    if premise_fault:
        confidence = 0.82 if not hedged else 0.72
        return "SUPPORT", confidence, {"SUPPORT": confidence, "UNCERTAIN": 1.0 - confidence}
    if premise_weak_absence and claims_absence and overconfident:
        return "OVERCLAIM", 0.76, {"OVERCLAIM": 0.76, "UNCERTAIN": 0.18, "SUPPORT": 0.06}
    if premise_weak_absence and claims_absence:
        return "SUPPORT", 0.62, {"SUPPORT": 0.62, "UNCERTAIN": 0.30, "OVERCLAIM": 0.08}
    return "UNCERTAIN", 0.68, {"UNCERTAIN": 0.68, "SUPPORT": 0.20, "CONTRADICT": 0.08, "OVERCLAIM": 0.04}


def load_model_verifier(model_path: Path, max_length: int):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    def verify_pair(premise: str, hypothesis: str) -> tuple[str, float, dict[str, float]]:
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()
        label_map = {idx: ID_TO_VERIFIER_LABEL.get(idx, f"LABEL_{idx}") for idx in range(len(probs))}
        pred_id = int(torch.argmax(logits).item())
        scores = {label_map[idx]: float(probs[idx]) for idx in range(len(probs))}
        return label_map[pred_id], float(probs[pred_id]), scores

    return verify_pair


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
