from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}

K2_REPO_ID = "daven3/k2"
K2_MODEL_DIR = Path("models/k2")
K2_TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"
VISION_MODEL_NAME = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
VISION_ADAPTER_DIR = Path("outputs/vision_llm_trained/final")
VISION_PREFIX_PROJECTOR = Path("outputs/k2_qwen_vision_projector.pt")
K2_VISION_OUTPUT_DIR = Path("outputs/k2_attached_vision")
DEFAULT_SPLIT_DIR = Path("data/splits")
DEFAULT_TRAIN_DATA = DEFAULT_SPLIT_DIR / "train.csv"
DEFAULT_EVAL_DATA = DEFAULT_SPLIT_DIR / "validate.csv"
DEFAULT_TEST_DATA = DEFAULT_SPLIT_DIR / "test.csv"
K2_FINAL_DIR = K2_VISION_OUTPUT_DIR / "final"
K2_TRAINED_VISION_ADAPTER_DIR = K2_FINAL_DIR / "qwen_vision_adapter"
K2_TRAINED_PROJECTOR = K2_FINAL_DIR / "k2_qwen_vision_projector.pt"
K2_TRAINED_LORA_DIR = K2_FINAL_DIR / "k2_lora_adapter"
NUM_VISION_PREFIX_TOKENS = 8
VISION_TOKEN_DROP_RATE = 0.75
