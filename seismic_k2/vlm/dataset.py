import csv
import json
import random
from pathlib import Path

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    Dataset = object


def resize_image_for_training(image, max_side):
    if max_side is None or max_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1))
    from PIL import Image

    return image.resize(new_size, Image.Resampling.BICUBIC)


def load_exported_records(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            normalize_record(json.loads(line), path.parent)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return [normalize_record(row, path.parent) for row in csv.DictReader(file)]
    raise ValueError(f"unsupported dataset format: {path}. Expected .csv or .jsonl")


def split_exported_csv(
    input_csv,
    output_dir,
    train_ratio=0.8,
    validate_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    if input_csv.suffix.lower() != ".csv":
        raise ValueError(f"split currently expects the CSV export, got: {input_csv}")
    if min(train_ratio, validate_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    ratio_sum = train_ratio + validate_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("at least one split ratio must be greater than zero")

    with input_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"CSV has no header: {input_csv}")
    if not rows:
        raise ValueError(f"CSV has no rows: {input_csv}")

    grouped = {}
    for row in rows:
        key = row.get("source_row_id") or row.get("sample_id") or row.get("id") or row.get("row_id")
        grouped.setdefault(key, []).append(row)

    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)

    train_ratio /= ratio_sum
    validate_ratio /= ratio_sum
    train_target = int(round(len(groups) * train_ratio))
    validate_target = int(round(len(groups) * validate_ratio))
    train_groups = groups[:train_target]
    validate_groups = groups[train_target : train_target + validate_target]
    test_groups = groups[train_target + validate_target :]

    splits = {
        "train": flatten_groups(train_groups),
        "validate": flatten_groups(validate_groups),
        "test": flatten_groups(test_groups),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_name, split_rows in splits.items():
        path = output_dir / f"{split_name}.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        paths[split_name] = path

    return {
        "input_csv": input_csv.as_posix(),
        "output_dir": output_dir.as_posix(),
        "seed": seed,
        "groups": len(groups),
        "rows": len(rows),
        "splits": {
            name: {
                "path": paths[name].as_posix(),
                "rows": len(split_rows),
            }
            for name, split_rows in splits.items()
        },
    }


def flatten_groups(groups):
    return [row for group in groups for row in group]


def normalize_record(row, dataset_dir):
    row = dict(row)
    if "messages_json" in row and row.get("messages_json") and "messages" not in row:
        row["messages"] = json.loads(row["messages_json"])
    if "metadata_json" in row and row.get("metadata_json") and "metadata" not in row:
        row["metadata"] = json.loads(row["metadata_json"])

    image = row.get("image") or image_from_messages(row.get("messages", []))
    if not image:
        raise ValueError(f"record {row.get('id') or row.get('row_id') or '<unknown>'} has no image")
    row["image"] = resolve_dataset_path(image, dataset_dir).as_posix()
    mask_image = row.get("mask_image") or row.get("mask")
    if mask_image:
        row["mask_image"] = resolve_dataset_path(mask_image, dataset_dir).as_posix()

    row["question"] = row.get("question") or row.get("instruction") or question_from_messages(row.get("messages", []))
    row["answer"] = row.get("answer") or assistant_text_from_messages(row.get("messages", []))
    if not row["question"]:
        raise ValueError(f"record {row.get('id') or row.get('row_id') or '<unknown>'} has no question/instruction")
    if not row["answer"]:
        raise ValueError(f"record {row.get('id') or row.get('row_id') or '<unknown>'} has no answer")
    return row


def resolve_dataset_path(value, dataset_dir):
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return dataset_dir / path


def image_from_messages(messages):
    for message in messages or []:
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image" and item.get("image"):
                    return item["image"]
    return ""


def question_from_messages(messages):
    for message in messages or []:
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        texts = [item.get("text", "") for item in content if item.get("type") == "text"]
        return "\n".join(texts).strip()
    return ""


def assistant_text_from_messages(messages):
    for message in messages or []:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        for item in content:
            if item.get("type") == "text":
                return item.get("text", "")
    return ""


class ExportedMultimodalDataset(Dataset):
    def __init__(self, path, max_image_side=512, mask_output_size=256):
        self.path = Path(path)
        self.max_image_side = max_image_side
        self.mask_output_size = mask_output_size
        self.records = load_exported_records(self.path)
        if not self.records:
            raise ValueError(f"no records found in {self.path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        import torch

        record = self.records[index]
        image_path = Path(record["image"])
        image = Image.open(image_path).convert("RGB")
        image = resize_image_for_training(image, self.max_image_side)
        mask = torch.zeros((1, self.mask_output_size, self.mask_output_size), dtype=torch.float32)
        mask_valid = torch.zeros(1, dtype=torch.float32)
        if record.get("mask_image") and Path(record["mask_image"]).exists():
            mask = load_mask_image(record["mask_image"], self.mask_output_size)
            mask_valid = torch.ones(1, dtype=torch.float32)
        return {
            "id": record.get("id") or record.get("row_id") or str(index),
            "image": image,
            "question": record["question"],
            "answer": record["answer"],
            "mask": mask,
            "mask_valid": mask_valid,
            "mask_image": record.get("mask_image", ""),
        }


def load_mask_image(path, output_size):
    from PIL import Image
    import numpy as np
    import torch

    image = Image.open(path).convert("L")
    image = image.resize((output_size, output_size), Image.Resampling.NEAREST)
    mask = (np.asarray(image) > 0).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)
