import json
import os
from pathlib import Path

from seismic_k2.vlm.dataset import split_exported_csv


DATASET_CSV = Path(
    os.getenv(
        "DATASET_CSV",
        "/home/third/Desktop/simulationv2/Dataset/multimodal_verified_dataset.csv",
    )
)
SPLIT_DIR = Path(os.getenv("SPLIT_DIR", "data/splits"))
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.8"))
VALIDATE_RATIO = float(os.getenv("VALIDATE_RATIO", "0.1"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.1"))
SEED = int(os.getenv("SPLIT_SEED", "42"))


def split_dataset():
    result = split_exported_csv(
        DATASET_CSV,
        SPLIT_DIR,
        train_ratio=TRAIN_RATIO,
        validate_ratio=VALIDATE_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    split_dataset()
