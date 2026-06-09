# Seismic K2 Vision Training

This repository trains K2 with a Qwen vision encoder on the completed seismic multimodal dataset from:

`/home/third/Desktop/simulationv2/Dataset/multimodal_verified_dataset.csv`

Architecture diagram: [docs/architecture.svg](docs/architecture.svg)

## Workflow

Run the whole default workflow:

```bash
uv run python main.py
```

`main.py` does three things:

1. Splits the completed CSV into `data/splits/train.csv`, `data/splits/validate.csv`, and `data/splits/test.csv`.
2. Checks that K2 exists in `models/k2`, downloading it with Hugging Face `snapshot_download` only if the local snapshot is incomplete.
3. Starts training with W&B metrics enabled.

The default training run is intentionally simple:

- K2 is loaded in 4-bit quantization.
- K2 is trained with LoRA through PEFT.
- Qwen-VL vision layers are trained with LoRA through PEFT.
- A small projector learns to turn Qwen image features into K2 prefix embeddings.
- Training uses a normal PyTorch `DataLoader`, `AdamW`, and a manual loop.

The segmentation head is disabled by default. That matches the current dataset state: the CSV has `mask_image` and `overlay_image` columns, but the rows are currently empty, so there is no mask supervision yet.

## Separate Jobs

Download K2 only:

```bash
uv run python scripts/download_k2.py
```

Split the dataset only:

```bash
uv run python scripts/split_dataset.py
```

Run inference on one image:

```bash
IMAGE=/path/to/seismic_slice.png \
QUESTION="What seismic features are visible in this slice?" \
uv run python -m seismic_k2.vlm.sample_inference
```

Train the separate InternVL version:

```bash
uv run python scripts/train_internvl.py
```

Useful InternVL environment variables:

```bash
INTERNVL_MODEL_ID=OpenGVLab/InternVL3-2B
INTERNVL_OUTPUT_DIR=outputs/internvl_seismic
TRAIN_MASK_DECODER=0
MAX_IMAGE_SIDE=512
```

## Simple Configuration

Use environment variables instead of command-line parsers.

Dataset split:

```bash
DATASET_CSV=/home/third/Desktop/simulationv2/Dataset/multimodal_verified_dataset.csv
SPLIT_DIR=data/splits
TRAIN_RATIO=0.8
VALIDATE_RATIO=0.1
TEST_RATIO=0.1
SPLIT_SEED=42
```

K2 download:

```bash
K2_MODEL_DIR=models/k2
K2_REPO_ID=daven3/k2
K2_REVISION=
```

Training:

```bash
OUTPUT_DIR=outputs/k2_attached_vision
TRAIN_DATA=data/splits/train.csv
EVAL_DATA=data/splits/validate.csv
EPOCHS=1
TRAIN_QWEN_VISION=1
MAX_IMAGE_SIDE=512
WANDB_PROJECT=k2-seismic-lisa
WANDB_MODE=online
```

## Dataset Fields

The loader uses:

- `image`
- `question` or `instruction`
- `answer`
- `mask_image`, when present and non-empty
- `messages` or `messages_json` as fallback

`overlay_image` is treated as a visualization artifact. The model predicts masks from `mask_image` supervision when those paths are available, then renders overlays from predicted masks.
