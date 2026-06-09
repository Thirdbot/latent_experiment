import os
from pathlib import Path

os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from scripts.download_k2 import download_k2_snapshot
from scripts.split_dataset import split_dataset


K2_DIR = Path(os.getenv("K2_MODEL_DIR", "models/k2"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/k2_attached_vision"))
TRAIN_DATA = Path(os.getenv("TRAIN_DATA", "data/splits/train.csv"))
EVAL_DATA = Path(os.getenv("EVAL_DATA", "data/splits/validate.csv"))
EPOCHS = float(os.getenv("EPOCHS", "1"))
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "k2-seismic-lisa")
WANDB_MODE = os.getenv("WANDB_MODE", "online")
DO_EVAL = os.getenv("DO_EVAL", "1") == "1"
TRAIN_QWEN_VISION = os.getenv("TRAIN_QWEN_VISION", "1") == "1"
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "512"))


def main():
    split_dataset()
    download_k2_snapshot(model_dir=K2_DIR)

    from seismic_k2.vlm.k2_vision import train_attached_vision

    train_attached_vision(
        k2_dir=K2_DIR,
        output_dir=OUTPUT_DIR,
        train_data=TRAIN_DATA,
        eval_data=EVAL_DATA,
        epochs=EPOCHS,
        do_eval=DO_EVAL,
        train_qwen_vision_adapter=TRAIN_QWEN_VISION,
        train_segmentation_head=False,
        max_image_side=MAX_IMAGE_SIDE,
        wandb_project=WANDB_PROJECT,
        wandb_mode=WANDB_MODE,
    )


if __name__ == "__main__":
    main()
