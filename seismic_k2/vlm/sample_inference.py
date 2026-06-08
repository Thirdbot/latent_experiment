import os
from pathlib import Path

from seismic_k2.config import IMAGE_SUFFIXES


DEFAULT_TRAINED_DIR = Path("outputs/k2_attached_vision/final")


def iter_images(root):
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def main():
    from seismic_k2.vlm.k2_vision import run_pipeline

    run_pipeline(
        image_path=Path(os.environ["IMAGE"]),
        trained_dir=Path(os.getenv("TRAINED_DIR", DEFAULT_TRAINED_DIR.as_posix())),
        vision_token_drop_rate=0.0,
        question=os.getenv("QUESTION"),
    )


if __name__ == "__main__":
    main()
