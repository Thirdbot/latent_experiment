import os
from pathlib import Path

from seismic_k2.config import IMAGE_SUFFIXES


DEFAULT_TRAINED_DIR = Path("outputs/k2_attached_vision/final")
DEFAULT_MASK_OUTPUT_SIZE = 256


def iter_images(root):
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def main():
    from seismic_k2.vlm.k2_vision import run_pipeline

    mask_output = os.getenv("MASK_OUTPUT")
    overlay_output = os.getenv("OVERLAY_OUTPUT")
    segmentation_head = os.getenv("SEGMENTATION_HEAD")

    run_pipeline(
        image_path=Path(os.environ["IMAGE"]),
        k2_dir=Path(os.getenv("K2_MODEL_DIR", "models/k2")),
        trained_dir=Path(os.getenv("TRAINED_DIR", DEFAULT_TRAINED_DIR.as_posix())),
        vision_token_drop_rate=0.0,
        question=os.getenv("QUESTION"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "128")),
        segmentation_head_path=Path(segmentation_head) if segmentation_head else None,
        mask_output=Path(mask_output) if mask_output else None,
        overlay_output=Path(overlay_output) if overlay_output else None,
        overlay_threshold=float(os.getenv("OVERLAY_THRESHOLD", "0.5")),
        mask_output_size=int(os.getenv("MASK_OUTPUT_SIZE", str(DEFAULT_MASK_OUTPUT_SIZE))),
    )


if __name__ == "__main__":
    main()
