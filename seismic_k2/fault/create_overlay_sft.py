from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT = "Analyze the seismic image and identify whether fault evidence is present."
POSITIVE_ANSWER = "The auxiliary fault overlay head is supervised with a ground-truth fault mask for this image."
NEGATIVE_ANSWER = "The auxiliary fault overlay head is supervised with an empty ground-truth fault mask for this image."


def iter_images(image_dir: Path):
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in suffixes:
            yield path


def label_path_for(data_root: Path, split: str, image_path: Path) -> Path:
    return data_root / "labels" / split / f"{image_path.stem}.txt"


def label_has_fault(label_path: Path) -> bool:
    return label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())


def make_record(data_root: Path, split: str, image_path: Path) -> dict:
    label_path = label_path_for(data_root, split, image_path)
    has_fault = label_has_fault(label_path)
    answer = POSITIVE_ANSWER if has_fault else NEGATIVE_ANSWER
    return {
        "image": image_path.as_posix(),
        "split": split,
        "question": PROMPT,
        "answer": answer,
        "fault_label_path": label_path.as_posix(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.as_posix()},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
    }


def write_split(data_root: Path, output_dir: Path, split: str) -> int:
    image_dir = data_root / "images" / split
    if not image_dir.exists():
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_sft.jsonl"
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in iter_images(image_dir):
            label_path = label_path_for(data_root, split, image_path)
            if not label_path.exists():
                continue
            handle.write(json.dumps(make_record(data_root, split, image_path), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SFT JSONL with real fault overlay labels from YOLO segmentation data.")
    parser.add_argument("--data-root", default="data/fault_yolo", help="YOLO dataset root with images/ and labels/.")
    parser.add_argument("--output-dir", default="outputs/fault_overlay_sft", help="Output directory for *_sft.jsonl files.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    summary = {}
    for split in args.splits:
        summary[split] = write_split(data_root, output_dir, split)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
