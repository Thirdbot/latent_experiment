from pathlib import Path


DEFAULT_VLLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OUTPUT_DIR = Path("outputs/generated_unicamp_instructions")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
VLLM_MAX_IMAGE_SIDE = 1280

K2_REPO_ID = "daven3/k2"
K2_MODEL_DIR = Path("models/k2")
K2_TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"
VISION_MODEL_NAME = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
VISION_ADAPTER_DIR = Path("outputs/vision_llm_trained/final")
VISION_PREFIX_PROJECTOR = Path("outputs/k2_qwen_vision_projector.pt")
K2_VISION_OUTPUT_DIR = Path("outputs/k2_attached_vision")
GENERATED_DATA_DIR = Path("outputs/generated_unicamp_instructions")
DEFAULT_TRAIN_JSONL = GENERATED_DATA_DIR / "train_sft.jsonl"
DEFAULT_EVAL_JSONL = GENERATED_DATA_DIR / "validation_sft.jsonl"
K2_FINAL_DIR = K2_VISION_OUTPUT_DIR / "final"
K2_TRAINED_VISION_ADAPTER_DIR = K2_FINAL_DIR / "qwen_vision_adapter"
K2_TRAINED_PROJECTOR = K2_FINAL_DIR / "k2_qwen_vision_projector.pt"
K2_TRAINED_LORA_DIR = K2_FINAL_DIR / "k2_lora_adapter"
NUM_VISION_PREFIX_TOKENS = 8
VISION_TOKEN_DROP_RATE = 0.75

