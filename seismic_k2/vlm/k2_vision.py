import argparse
import json
from pathlib import Path

import unsloth
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, load_peft_weights, set_peft_model_state_dict
from PIL import Image
from torch import nn
from unsloth import FastVisionModel, is_bfloat16_supported
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from seismic_k2.config import (
    DEFAULT_EVAL_JSONL,
    DEFAULT_TRAIN_JSONL,
    K2_FINAL_DIR,
    K2_MODEL_DIR,
    K2_REPO_ID,
    K2_TOKENIZER_NAME,
    K2_TRAINED_LORA_DIR,
    K2_TRAINED_PROJECTOR,
    K2_TRAINED_VISION_ADAPTER_DIR,
    K2_VISION_OUTPUT_DIR,
    NUM_VISION_PREFIX_TOKENS,
    VISION_ADAPTER_DIR,
    VISION_MODEL_NAME,
    VISION_PREFIX_PROJECTOR,
    VISION_TOKEN_DROP_RATE,
)
from seismic_k2.dataset_generator.prompts import IMAGE_PROMPT, K2_PROMPT, K2_VQA_PROMPT_TEMPLATE


FAULT_OVERLAY_SIZE = 32
FAULT_OVERLAY_LOSS_WEIGHT = 0.25
FAULT_PRESENCE_LOSS_WEIGHT = 0.10
DEFAULT_MAX_IMAGE_SIDE = 512
REAL_MASK_FIELDS = (
    "fault_mask_path",
    "mask_path",
    "segmentation_mask_path",
    "fault_label_path",
    "yolo_label_path",
)


def latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    checkpoint = get_last_checkpoint(output_dir.as_posix())
    if checkpoint is None:
        return None

    checkpoint = Path(checkpoint)
    loadable_files = (
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "k2_qwen_vision_projector.pt",
        "fault_overlay_head.pt",
    )
    if any((checkpoint / filename).exists() for filename in loadable_files):
        return checkpoint.as_posix()

    adapter_paths = (
        checkpoint / "qwen_vision_adapter" / "adapter_model.safetensors",
        checkpoint / "qwen_vision_adapter" / "adapter_model.bin",
        checkpoint / "k2_lora_adapter" / "adapter_model.safetensors",
        checkpoint / "k2_lora_adapter" / "adapter_model.bin",
    )
    if any(path.exists() for path in adapter_paths):
        return checkpoint.as_posix()

    print(f"ignoring checkpoint without loadable K2 vision files: {checkpoint}")
    return None


def has_k2_vision_checkpoint_parts(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    paths = (
        checkpoint_dir / "k2_qwen_vision_projector.pt",
        checkpoint_dir / "fault_overlay_head.pt",
        checkpoint_dir / "qwen_vision_adapter" / "adapter_model.safetensors",
        checkpoint_dir / "qwen_vision_adapter" / "adapter_model.bin",
        checkpoint_dir / "k2_lora_adapter" / "adapter_model.safetensors",
        checkpoint_dir / "k2_lora_adapter" / "adapter_model.bin",
    )
    return any(path.exists() for path in paths)


def final_dir_for(output_dir):
    return Path(output_dir) / "final"


def adapter_dir_for(output_dir):
    return final_dir_for(output_dir) / "qwen_vision_adapter"


def projector_path_for(output_dir):
    return final_dir_for(output_dir) / "k2_qwen_vision_projector.pt"


def k2_lora_dir_for(output_dir):
    return final_dir_for(output_dir) / "k2_lora_adapter"


def fault_overlay_head_path_for(output_dir):
    return final_dir_for(output_dir) / "fault_overlay_head.pt"


def resolve_existing_path(*paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    for path in paths:
        if path is not None:
            return Path(path)
    return None


def k2_repo_is_complete(model_dir):
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.exists():
        return False
    if not index_path.exists():
        return any(model_dir.glob("*.bin")) or any(model_dir.glob("*.safetensors"))

    with index_path.open("r", encoding="utf-8") as file:
        index = json.load(file)
    weight_files = set(index.get("weight_map", {}).values())
    return bool(weight_files) and all((model_dir / filename).exists() for filename in weight_files)


def download_k2(model_dir=K2_MODEL_DIR):
    model_dir = Path(model_dir)
    if k2_repo_is_complete(model_dir):
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {K2_REPO_ID} to {model_dir}")
    snapshot_download(
        repo_id=K2_REPO_ID,
        local_dir=model_dir.as_posix(),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return model_dir


def has_tokenizer_files(path):
    path = Path(path)
    return any(
        (path / filename).exists()
        for filename in (
            "tokenizer.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
    )


def load_k2_tokenizer(model_dir=K2_MODEL_DIR, tokenizer_name=K2_TOKENIZER_NAME):
    tokenizer = None
    if has_tokenizer_files(model_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                Path(model_dir).as_posix(),
                use_fast=False,
                trust_remote_code=True,
            )
        except Exception:
            tokenizer = None

    if tokenizer is None or getattr(tokenizer, "vocab_size", 0) < 1000:
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
        )

    if getattr(tokenizer, "vocab_size", 0) < 1000:
        raise ValueError(
            f"Invalid K2 tokenizer vocab_size={getattr(tokenizer, 'vocab_size', None)}. "
            "K2 needs a LLaMA tokenizer, not the incomplete local tokenizer files."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_k2(model_dir=K2_MODEL_DIR, tokenizer_name=K2_TOKENIZER_NAME, lora_adapter_dir=None, is_trainable=False):
    model_dir = download_k2(model_dir)
    tokenizer = load_k2_tokenizer(model_dir, tokenizer_name)
    print(f"loaded K2 tokenizer vocab_size={tokenizer.vocab_size}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir.as_posix(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if lora_adapter_dir is not None and Path(lora_adapter_dir).exists():
        model = PeftModel.from_pretrained(
            model,
            Path(lora_adapter_dir).as_posix(),
            is_trainable=is_trainable,
        )
        print(f"loaded K2 LoRA adapter from {lora_adapter_dir}")
    model.eval()
    return model, tokenizer


def add_k2_lora(model, r=8, alpha=16, dropout=0.05):
    if isinstance(model, PeftModel):
        return model
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"K2 LoRA trainable params: {trainable_params:,} / {total_params:,}")
    return model


def load_qwen_vision_encoder(trainable=False, adapter_dir=None):
    device_map = {"": 0} if torch.cuda.is_available() else None
    model, processor = FastVisionModel.from_pretrained(
        VISION_MODEL_NAME,
        load_in_4bit=True,
        device_map=device_map,
        use_gradient_checkpointing="unsloth",
    )

    if adapter_dir is None:
        adapter_dir = resolve_existing_path(K2_TRAINED_VISION_ADAPTER_DIR, VISION_ADAPTER_DIR)
    else:
        adapter_dir = Path(adapter_dir)
    has_vision_adapter = adapter_dir.exists()
    if has_vision_adapter:
        model = PeftModel.from_pretrained(
            model,
            adapter_dir.as_posix(),
            is_trainable=trainable,
        )
        if trainable:
            print(f"loaded trainable Qwen vision adapter from {adapter_dir}")
        else:
            model = model.merge_and_unload()
            print(f"merged Qwen vision adapter from {adapter_dir}")
    else:
        print(f"vision adapter not found at {adapter_dir}; using base Qwen-VL")

    if trainable:
        if not isinstance(model, PeftModel):
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=True,
                finetune_language_layers=False,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in model.parameters())
        print(f"Qwen trainable params: {trainable_params:,} / {total_params:,}")
        model.train()
    else:
        model.eval()
    return model, processor


def get_hidden_size(model):
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None) is not None:
        return text_config.hidden_size
    if config is not None and getattr(config, "hidden_size", None) is not None:
        return config.hidden_size
    raise ValueError("Could not infer hidden_size.")


def get_image_token_id(model, processor):
    config = getattr(model, "config", None)
    for attr in ("image_token_id", "image_token_index"):
        value = getattr(config, attr, None)
        if value is not None:
            return value

    tokenizer = getattr(processor, "tokenizer", processor)
    for token in ("<|image_pad|>", "<image>", "<|vision_start|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return token_id

    raise ValueError("Could not infer Qwen image token id.")


def apply_image_token_dropout(image_mask, drop_rate=VISION_TOKEN_DROP_RATE, training=False):
    if not training or drop_rate <= 0:
        return image_mask

    keep_prob = max(1.0 - drop_rate, 1e-4)
    keep_mask = torch.rand(image_mask.shape, device=image_mask.device) < keep_prob
    keep_mask = keep_mask & image_mask

    empty_rows = image_mask.any(dim=1) & ~keep_mask.any(dim=1)
    if empty_rows.any():
        row_indices = empty_rows.nonzero(as_tuple=False).flatten()
        for row in row_indices.tolist():
            token_indices = image_mask[row].nonzero(as_tuple=False).flatten()
            selected = token_indices[torch.randint(token_indices.numel(), (1,), device=image_mask.device)]
            keep_mask[row, selected] = True
    return keep_mask


class VisionPrefixProjector(nn.Module):
    def __init__(self, vision_hidden_size, k2_hidden_size, num_prefix_tokens=NUM_VISION_PREFIX_TOKENS):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.net = nn.Sequential(
            nn.Linear(vision_hidden_size, k2_hidden_size),
            nn.GELU(),
            nn.Linear(k2_hidden_size, k2_hidden_size * num_prefix_tokens),
        )

    def forward(self, vision_latent):
        prefix = self.net(vision_latent)
        return prefix.view(vision_latent.size(0), self.num_prefix_tokens, -1)


def load_or_create_projector(vision_hidden_size, k2_hidden_size, projector_path=VISION_PREFIX_PROJECTOR):
    projector = VisionPrefixProjector(vision_hidden_size, k2_hidden_size)
    projector_path = Path(projector_path)
    if projector_path.exists():
        state = torch.load(projector_path, map_location="cpu")
        projector.load_state_dict(state)
        print(f"loaded vision-prefix projector from {projector_path}")
    else:
        print(
            f"projector not found at {projector_path}; using randomly initialized projector. "
            "Train this projector before expecting grounded K2 image understanding."
        )
    return projector


def encode_image_with_qwen(qwen_model, qwen_processor, image, token_drop_rate=0.0):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": IMAGE_PROMPT},
            ],
        }
    ]
    prompt = qwen_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = qwen_processor(
        text=[prompt],
        images=[[image]],
        return_tensors="pt",
    )
    device = next(qwen_model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    image_token_id = get_image_token_id(qwen_model, qwen_processor)

    with torch.no_grad():
        outputs = qwen_forward_hidden_states(qwen_model, inputs)

    hidden_states = outputs.hidden_states[-1]
    image_mask = inputs["input_ids"].eq(image_token_id)
    image_mask = apply_image_token_dropout(
        image_mask,
        drop_rate=token_drop_rate,
        training=token_drop_rate > 0,
    )
    mask = image_mask.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def encode_qwen_inputs(qwen_model, qwen_processor, inputs, token_drop_rate=VISION_TOKEN_DROP_RATE, return_tokens=False):
    image_token_id = get_image_token_id(qwen_model, qwen_processor)
    outputs = qwen_forward_hidden_states(qwen_model, inputs)
    hidden_states = outputs.hidden_states[-1]
    image_mask = inputs["input_ids"].eq(image_token_id)
    image_mask = apply_image_token_dropout(
        image_mask,
        drop_rate=token_drop_rate,
        training=qwen_model.training,
    )
    mask = image_mask.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    vision_latent = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if not return_tokens:
        return vision_latent
    return vision_latent, hidden_states, image_mask


def qwen_forward_hidden_states(qwen_model, inputs):
    kwargs = {
        **inputs,
        "output_hidden_states": True,
        "return_dict": True,
        "use_cache": False,
        # Qwen-VL otherwise computes full vocabulary logits even though this
        # path only needs hidden states. Keeping one logit position avoids a
        # multi-GB lm_head allocation on 24GB GPUs.
        "logits_to_keep": 1,
    }
    try:
        return qwen_model(**kwargs)
    except TypeError:
        kwargs.pop("logits_to_keep", None)
        return qwen_model(**kwargs)


def generate_with_visual_prefix(k2_model, k2_tokenizer, visual_prefix, prompt):
    device = k2_model.get_input_embeddings().weight.device
    text_inputs = k2_tokenizer(prompt, return_tensors="pt")
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    text_embeddings = k2_model.get_input_embeddings()(text_inputs["input_ids"])
    visual_prefix = visual_prefix.to(device=device, dtype=text_embeddings.dtype)
    inputs_embeds = torch.cat([visual_prefix, text_embeddings], dim=1)
    prefix_mask = torch.ones(
        visual_prefix.size()[:2],
        dtype=text_inputs["attention_mask"].dtype,
        device=device,
    )
    attention_mask = torch.cat([prefix_mask, text_inputs["attention_mask"]], dim=1)

    generation_kwargs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "max_new_tokens": 128,
        "min_new_tokens": 8,
        "do_sample": False,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
        "pad_token_id": k2_tokenizer.pad_token_id,
    }
    if k2_tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = k2_tokenizer.eos_token_id

    with torch.no_grad():
        generated_ids = k2_model.generate(**generation_kwargs)
    token_ids = generated_ids[0].detach().cpu().tolist()
    token_variants = [token_ids]
    if len(token_ids) > 1:
        token_variants.append(token_ids[1:])
    special_ids = {
        token_id
        for token_id in (
            getattr(k2_tokenizer, "bos_token_id", None),
            getattr(k2_tokenizer, "eos_token_id", None),
            getattr(k2_tokenizer, "pad_token_id", None),
            getattr(k2_tokenizer, "unk_token_id", None),
        )
        if token_id is not None
    }
    filtered_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    if filtered_ids and filtered_ids != token_ids:
        token_variants.append(filtered_ids)

    decode_candidates = [
        k2_tokenizer.decode(ids, skip_special_tokens=skip_special).strip()
        for ids in token_variants
        for skip_special in (True, False)
    ]
    sp_model = getattr(k2_tokenizer, "sp_model", None)
    if sp_model is not None:
        for ids in token_variants:
            try:
                decode_candidates.append(sp_model.DecodeIds([int(token_id) for token_id in ids]).strip())
            except Exception:
                pass

    text = ""
    for candidate in decode_candidates:
        candidate = candidate.strip()
        if not candidate or candidate == getattr(k2_tokenizer, "unk_token", "<unk>"):
            continue
        if candidate.startswith(getattr(k2_tokenizer, "unk_token", "<unk>")):
            candidate = candidate[len(getattr(k2_tokenizer, "unk_token", "<unk>")) :].strip()
        if not candidate:
            continue
        text = candidate
        break
    if "Answer:" in text:
        text = text.rsplit("Answer:", 1)[-1].strip()
    if text:
        return text

    raw_text = decode_candidates[-1] if decode_candidates else ""
    token_preview = token_ids[:64]
    return f"[empty decoded output] raw={raw_text!r} token_ids={token_preview}"


def build_k2_vqa_prompt(question):
    return K2_VQA_PROMPT_TEMPLATE.format(question=question)


def get_message_text(messages):
    if not messages:
        return ""
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        for item in content:
            if item.get("type") == "text":
                return item.get("text", "")
    return ""


def get_question(record):
    if record.get("question"):
        return record["question"]
    messages = record.get("messages", [])
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        texts = [item.get("text", "") for item in content if item.get("type") == "text"]
        return "\n".join(texts).strip()
    return K2_PROMPT


def resize_image_for_training(image, max_side=DEFAULT_MAX_IMAGE_SIDE):
    if max_side is None or max_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def resolve_record_path(record, value):
    path = Path(value)
    if path.exists():
        return path
    image_value = record.get("image")
    if image_value:
        image_parent = Path(image_value).parent
        candidate = image_parent / path
        if candidate.exists():
            return candidate
    return path


def find_yolo_label_for_image(image_path):
    image_path = Path(image_path)
    parts = list(image_path.parts)
    if "images" not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    return label_path if label_path.exists() else None


def real_fault_label_path(record):
    for field in REAL_MASK_FIELDS:
        value = record.get(field)
        if value:
            path = resolve_record_path(record, value)
            if path.exists():
                return path
    image_path = record.get("image")
    if image_path:
        return find_yolo_label_for_image(image_path)
    return None


def rasterize_fault_target(record, image_size, output_size=FAULT_OVERLAY_SIZE, allow_pseudo_labels=False):
    label_path = real_fault_label_path(record)
    if label_path is not None:
        suffix = label_path.suffix.lower()
        if suffix == ".txt":
            mask = rasterize_yolo_label(label_path, output_size=output_size)
        else:
            mask = load_fault_mask(label_path, output_size=output_size)
        return mask, torch.tensor([float(mask.max().item() > 0.5)], dtype=torch.float32), torch.tensor([1.0]), "real"

    if allow_pseudo_labels:
        detections = record.get("faultnet_detections") or []
        mask = rasterize_fault_overlay(detections, image_size, output_size=output_size)
        return mask, torch.tensor([1.0 if detections else 0.0], dtype=torch.float32), torch.tensor([1.0]), "pseudo"

    mask = torch.zeros((1, output_size, output_size), dtype=torch.float32)
    return mask, torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32), "missing"


def load_fault_mask(path, output_size=FAULT_OVERLAY_SIZE):
    path = Path(path)
    if path.suffix.lower() in {".npy", ".npz"}:
        array = np.load(path)
        if isinstance(array, np.lib.npyio.NpzFile):
            key = "fault" if "fault" in array.files else array.files[0]
            array = array[key]
        if array.ndim == 3:
            array = array[array.shape[0] // 2]
        image = Image.fromarray((array.astype(np.float32) > 0).astype(np.uint8) * 255, mode="L")
    else:
        image = Image.open(path).convert("L")
    image = image.resize((output_size, output_size), Image.Resampling.NEAREST)
    mask = (np.asarray(image) > 0).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)


def rasterize_yolo_label(path, output_size=FAULT_OVERLAY_SIZE):
    mask = torch.zeros((1, output_size, output_size), dtype=torch.float32)
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return mask
    for line in text.splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        coords = [float(value) for value in values[1:]]
        points = []
        for x, y in zip(coords[0::2], coords[1::2]):
            px = int(round(x * (output_size - 1)))
            py = int(round(y * (output_size - 1)))
            points.append((min(max(px, 0), output_size - 1), min(max(py, 0), output_size - 1)))
        if len(points) >= 3:
            fill_polygon(mask[0], points)
    return mask


def rasterize_fault_overlay(detections, image_size, output_size=FAULT_OVERLAY_SIZE):
    mask = torch.zeros((1, output_size, output_size), dtype=torch.float32)
    width, height = image_size
    if width <= 0 or height <= 0:
        return mask

    for detection in detections or []:
        polygon = detection.get("polygon_xy_sample") or []
        if len(polygon) >= 3:
            points = []
            for x, y in polygon:
                px = int(round(float(x) / width * (output_size - 1)))
                py = int(round(float(y) / height * (output_size - 1)))
                points.append((min(max(px, 0), output_size - 1), min(max(py, 0), output_size - 1)))
            fill_polygon(mask[0], points)
            continue

        box = detection.get("box_xyxy")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        x1 = int(round(x1 / width * (output_size - 1)))
        x2 = int(round(x2 / width * (output_size - 1)))
        y1 = int(round(y1 / height * (output_size - 1)))
        y2 = int(round(y2 / height * (output_size - 1)))
        x1, x2 = sorted((min(max(x1, 0), output_size - 1), min(max(x2, 0), output_size - 1)))
        y1, y2 = sorted((min(max(y1, 0), output_size - 1), min(max(y2, 0), output_size - 1)))
        mask[:, y1 : y2 + 1, x1 : x2 + 1] = 1.0
    return mask


def fill_polygon(mask, points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = max(min(xs), 0), min(max(xs), mask.shape[1] - 1)
    min_y, max_y = max(min(ys), 0), min(max(ys), mask.shape[0] - 1)
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if point_in_polygon(x + 0.5, y + 0.5, points):
                mask[y, x] = 1.0


def point_in_polygon(x, y, points):
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-6) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


class K2VisionDataset(Dataset):
    def __init__(self, jsonl_path, allow_pseudo_fault_labels=False, max_image_side=DEFAULT_MAX_IMAGE_SIDE):
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Generated SFT JSONL not found: {jsonl_path}")
        self.allow_pseudo_fault_labels = allow_pseudo_fault_labels
        self.max_image_side = max_image_side
        self.records = []
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.records.append(json.loads(line))
        if not self.records:
            raise ValueError(f"No records found in {jsonl_path}")
        self.fault_label_source_counts = self.count_fault_label_sources()

    def count_fault_label_sources(self):
        counts = {"real": 0, "pseudo": 0, "missing": 0}
        for record in self.records:
            if real_fault_label_path(record) is not None:
                counts["real"] += 1
            elif self.allow_pseudo_fault_labels and "faultnet_detections" in record:
                counts["pseudo"] += 1
            else:
                counts["missing"] += 1
        return counts

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = Path(record["image"])
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        image = resize_image_for_training(image, self.max_image_side)
        answer = get_message_text(record.get("messages", [])) or record.get("chosen_answer", "")
        question = get_question(record)
        fault_overlay, fault_presence, fault_valid, fault_source = rasterize_fault_target(
            record,
            original_size,
            allow_pseudo_labels=self.allow_pseudo_fault_labels,
        )
        return {
            "image": image,
            "question": question,
            "answer": answer,
            "fault_overlay": fault_overlay,
            "fault_presence": fault_presence,
            "fault_overlay_valid": fault_valid,
            "fault_label_source": fault_source,
        }


class K2VisionCollator:
    def __init__(self, qwen_processor, k2_tokenizer, max_length=2048):
        self.qwen_processor = qwen_processor
        self.k2_tokenizer = k2_tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        qwen_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": IMAGE_PROMPT},
                    ],
                }
            ]
            for _ in examples
        ]
        qwen_prompts = [
            self.qwen_processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
            )
            for message in qwen_messages
        ]
        qwen_inputs = self.qwen_processor(
            text=qwen_prompts,
            images=[[example["image"]] for example in examples],
            return_tensors="pt",
            padding=True,
        )

        prompt_texts = [
            build_k2_vqa_prompt(example["question"])
            for example in examples
        ]
        input_id_rows = []
        label_rows = []
        attention_rows = []
        pad_token_id = self.k2_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.k2_tokenizer.eos_token_id

        for prompt_text, example in zip(prompt_texts, examples):
            prompt_ids = self.k2_tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            answer_ids = self.k2_tokenizer(
                f"\n{example['answer']}",
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            if not answer_ids:
                raise ValueError("Empty answer text produced no K2 target tokens.")

            if len(answer_ids) >= self.max_length:
                answer_ids = answer_ids[: self.max_length - 1]
            max_prompt_len = self.max_length - len(answer_ids)
            if max_prompt_len <= 0:
                raise ValueError(
                    "No room left for prompt tokens. Increase K2VisionCollator max_length "
                    "or shorten generated answers."
                )
            prompt_ids = prompt_ids[-max_prompt_len:]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids

            input_id_rows.append(input_ids)
            label_rows.append(labels)
            attention_rows.append([1] * len(input_ids))

        batch_length = max(len(row) for row in input_id_rows)
        for input_ids, labels, attention_mask in zip(input_id_rows, label_rows, attention_rows):
            pad_len = batch_length - len(input_ids)
            input_ids.extend([pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
            attention_mask.extend([0] * pad_len)

        k2_input_ids = torch.tensor(input_id_rows, dtype=torch.long)
        labels = torch.tensor(label_rows, dtype=torch.long)
        k2_attention_mask = torch.tensor(attention_rows, dtype=torch.long)
        if labels.ne(-100).sum().item() == 0:
            raise ValueError(
                "No answer tokens left for loss after masking. "
                "Increase K2VisionCollator max_length or shorten the prompt/answers."
            )

        batch = {f"qwen_{key}": value for key, value in qwen_inputs.items()}
        batch["k2_input_ids"] = k2_input_ids
        batch["k2_attention_mask"] = k2_attention_mask
        batch["labels"] = labels
        batch["fault_overlay"] = torch.stack([example["fault_overlay"] for example in examples])
        batch["fault_presence"] = torch.stack([example["fault_presence"] for example in examples])
        batch["fault_overlay_valid"] = torch.stack([example["fault_overlay_valid"] for example in examples])
        return batch


class FaultOverlayHead(nn.Module):
    def __init__(self, vision_hidden_size, output_size=FAULT_OVERLAY_SIZE):
        super().__init__()
        self.output_size = output_size
        hidden = max(vision_hidden_size // 4, 64)
        self.token_score = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.presence_head = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, vision_tokens, image_mask, vision_latent):
        batch_size = vision_tokens.size(0)
        token_scores = self.token_score(vision_tokens).squeeze(-1)
        token_scores = token_scores.masked_fill(~image_mask, 0.0)
        token_counts = image_mask.sum(dim=1).clamp_min(1)
        max_tokens = int(token_counts.max().item())
        side = max(int(max_tokens**0.5), 1)
        while side * side < max_tokens:
            side += 1

        grid = vision_tokens.new_zeros((batch_size, 1, side * side))
        for row in range(batch_size):
            values = token_scores[row, image_mask[row]]
            grid[row, 0, : values.numel()] = values
        grid = grid.view(batch_size, 1, side, side)
        overlay_logits = F.interpolate(
            grid,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        presence_logits = self.presence_head(vision_latent)
        return {
            "fault_overlay_logits": overlay_logits,
            "fault_presence_logits": presence_logits,
        }


def load_or_create_fault_overlay_head(vision_hidden_size, head_path=None, output_size=FAULT_OVERLAY_SIZE):
    head = FaultOverlayHead(vision_hidden_size, output_size=output_size)
    if head_path is not None and Path(head_path).exists():
        state = torch.load(head_path, map_location="cpu")
        head.load_state_dict(state)
        print(f"loaded fault overlay head from {head_path}")
    elif head_path is not None:
        print(f"fault overlay head not found at {head_path}; using randomly initialized head")
    return head


class FrozenK2VisionModel(nn.Module):
    def __init__(
        self,
        qwen_model,
        qwen_processor,
        k2_model,
        projector,
        fault_overlay_head=None,
        vision_token_drop_rate=VISION_TOKEN_DROP_RATE,
        fault_overlay_loss_weight=FAULT_OVERLAY_LOSS_WEIGHT,
        fault_presence_loss_weight=FAULT_PRESENCE_LOSS_WEIGHT,
    ):
        super().__init__()
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.k2_model = k2_model
        self.projector = projector
        self.fault_overlay_head = fault_overlay_head
        self.vision_token_drop_rate = vision_token_drop_rate
        self.fault_overlay_loss_weight = fault_overlay_loss_weight
        self.fault_presence_loss_weight = fault_presence_loss_weight
        self.qwen_trainable = any(param.requires_grad for param in self.qwen_model.parameters())
        if not self.qwen_trainable:
            self.qwen_model.eval()

        self.k2_model.eval()
        for name, param in self.k2_model.named_parameters():
            param.requires_grad = param.requires_grad and "lora_" in name

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.qwen_trainable:
            self.qwen_model.eval()
        self.k2_model.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.qwen_model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.qwen_model.gradient_checkpointing_enable()
            else:
                self.qwen_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )

    def gradient_checkpointing_disable(self):
        if hasattr(self.qwen_model, "gradient_checkpointing_disable"):
            self.qwen_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        if hasattr(self.qwen_model, "enable_input_require_grads"):
            self.qwen_model.enable_input_require_grads()

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self.qwen_model, PeftModel):
            self.qwen_model.save_pretrained(output_dir / "qwen_vision_adapter")
        else:
            print("skipping full Qwen save; qwen_model is not a PEFT adapter")
        if isinstance(self.k2_model, PeftModel):
            self.k2_model.save_pretrained(output_dir / "k2_lora_adapter")
        torch.save(self.projector.state_dict(), output_dir / "k2_qwen_vision_projector.pt")
        if self.fault_overlay_head is not None:
            torch.save(self.fault_overlay_head.state_dict(), output_dir / "fault_overlay_head.pt")

    def save_checkpoint_parts(self, output_dir):
        self.save_pretrained(output_dir)

    def load_checkpoint_parts(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        qwen_adapter_dir = checkpoint_dir / "qwen_vision_adapter"
        if isinstance(self.qwen_model, PeftModel) and qwen_adapter_dir.exists():
            state = load_peft_weights(qwen_adapter_dir.as_posix())
            set_peft_model_state_dict(self.qwen_model, state)
            print(f"resumed Qwen vision adapter from {qwen_adapter_dir}")
        k2_lora_dir = checkpoint_dir / "k2_lora_adapter"
        if isinstance(self.k2_model, PeftModel) and k2_lora_dir.exists():
            state = load_peft_weights(k2_lora_dir.as_posix())
            set_peft_model_state_dict(self.k2_model, state)
            print(f"resumed K2 LoRA adapter from {k2_lora_dir}")
        projector_path = checkpoint_dir / "k2_qwen_vision_projector.pt"
        if projector_path.exists():
            self.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
            print(f"resumed projector from {projector_path}")
        fault_head_path = checkpoint_dir / "fault_overlay_head.pt"
        if self.fault_overlay_head is not None and fault_head_path.exists():
            self.fault_overlay_head.load_state_dict(torch.load(fault_head_path, map_location="cpu"))
            print(f"resumed fault overlay head from {fault_head_path}")

    def forward(self, **batch):
        qwen_inputs = {
            key.removeprefix("qwen_"): value.to(next(self.qwen_model.parameters()).device)
            for key, value in batch.items()
            if key.startswith("qwen_")
        }
        if self.qwen_trainable:
            vision_latent, vision_tokens, image_mask = encode_qwen_inputs(
                self.qwen_model,
                self.qwen_processor,
                qwen_inputs,
                token_drop_rate=self.vision_token_drop_rate,
                return_tokens=True,
            )
        else:
            with torch.no_grad():
                vision_latent, vision_tokens, image_mask = encode_qwen_inputs(
                    self.qwen_model,
                    self.qwen_processor,
                    qwen_inputs,
                    token_drop_rate=self.vision_token_drop_rate,
                    return_tokens=True,
                )
        fault_outputs = {}
        if self.fault_overlay_head is not None:
            head_device = next(self.fault_overlay_head.parameters()).device
            fault_outputs = self.fault_overlay_head(
                vision_tokens.to(head_device),
                image_mask.to(head_device),
                vision_latent.to(head_device),
            )

        projector_device = next(self.projector.parameters()).device
        vision_latent = vision_latent.to(
            device=projector_device,
            dtype=next(self.projector.parameters()).dtype,
        )
        visual_prefix = self.projector(vision_latent)

        k2_device = self.k2_model.get_input_embeddings().weight.device
        input_ids = batch["k2_input_ids"].to(k2_device)
        attention_mask = batch["k2_attention_mask"].to(k2_device)
        labels = batch["labels"].to(k2_device)
        text_embeddings = self.k2_model.get_input_embeddings()(input_ids)
        visual_prefix = visual_prefix.to(device=k2_device, dtype=text_embeddings.dtype)
        inputs_embeds = torch.cat([visual_prefix, text_embeddings], dim=1)
        prefix_mask = torch.ones(
            visual_prefix.size()[:2],
            dtype=attention_mask.dtype,
            device=k2_device,
        )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        prefix_labels = torch.full(
            visual_prefix.size()[:2],
            -100,
            dtype=labels.dtype,
            device=k2_device,
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.k2_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        if self.fault_overlay_head is not None and "fault_overlay" in batch:
            target_overlay = batch["fault_overlay"].to(
                device=fault_outputs["fault_overlay_logits"].device,
                dtype=fault_outputs["fault_overlay_logits"].dtype,
            )
            target_presence = batch["fault_presence"].to(
                device=fault_outputs["fault_presence_logits"].device,
                dtype=fault_outputs["fault_presence_logits"].dtype,
            )
            target_valid = batch["fault_overlay_valid"].to(
                device=fault_outputs["fault_presence_logits"].device,
                dtype=fault_outputs["fault_presence_logits"].dtype,
            ).flatten()
            valid_indices = target_valid > 0.5
            if valid_indices.any():
                overlay_logits = fault_outputs["fault_overlay_logits"][valid_indices]
                presence_logits = fault_outputs["fault_presence_logits"][valid_indices]
                overlay_loss = F.binary_cross_entropy_with_logits(
                    overlay_logits,
                    target_overlay[valid_indices],
                )
                dice = dice_loss_from_logits(overlay_logits, target_overlay[valid_indices])
                presence_loss = F.binary_cross_entropy_with_logits(
                    presence_logits,
                    target_presence[valid_indices],
                )
                outputs.loss = (
                    outputs.loss
                    + self.fault_overlay_loss_weight * (overlay_loss + dice)
                    + self.fault_presence_loss_weight * presence_loss
                )
                outputs.fault_overlay_loss = overlay_loss.detach()
                outputs.fault_overlay_dice_loss = dice.detach()
                outputs.fault_presence_loss = presence_loss.detach()
            outputs.fault_overlay_supervised = target_valid.sum().detach()
        for key, value in fault_outputs.items():
            setattr(outputs, key, value)
        return outputs


def dice_loss_from_logits(logits, target):
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3))
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def save_fault_overlay_prediction(image, overlay_prob, output_path, threshold=0.5, alpha=0.45):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = image.convert("RGB")
    mask = overlay_prob.detach().float().cpu().squeeze().clamp(0, 1).numpy()
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(base.size, Image.Resampling.BILINEAR)

    base_array = np.asarray(base).astype(np.float32)
    mask_array = np.asarray(mask_image).astype(np.float32) / 255.0
    hard_mask = mask_array >= threshold

    heat = np.zeros_like(base_array)
    heat[..., 0] = 255.0
    heat[..., 1] = 40.0
    heat[..., 2] = 40.0
    blend_weight = (mask_array * alpha)[..., None]
    overlay = base_array * (1.0 - blend_weight) + heat * blend_weight
    overlay[hard_mask] = overlay[hard_mask] * 0.65 + heat[hard_mask] * 0.35

    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)
    return output_path


class K2VisionTrainer(Trainer):
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target_model = model or self.model
        try:
            result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        except ValueError as error:
            if not has_k2_vision_checkpoint_parts(resume_from_checkpoint):
                raise
            print(f"skipping standard HF checkpoint model load: {error}")
            result = None
        if hasattr(target_model, "load_checkpoint_parts"):
            target_model.load_checkpoint_parts(resume_from_checkpoint)
        return result

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = Path(self._get_output_dir(trial=trial)) / checkpoint_folder
        if hasattr(model, "save_checkpoint_parts"):
            model.save_checkpoint_parts(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss.detach()
        return loss, None, None

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        self.model.save_pretrained(output_dir)


def save_training_history(log_history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(log_history, file, indent=2)

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    for item in log_history:
        step = item.get("step")
        if step is None:
            continue
        if "loss" in item:
            train_steps.append(step)
            train_losses.append(float(item["loss"]))
        if "eval_loss" in item:
            eval_steps.append(step)
            eval_losses.append(float(item["eval_loss"]))

    if not train_losses and not eval_losses:
        print(f"saved training history to {history_path}; no loss points found for plot")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"saved training history to {history_path}; install matplotlib to save loss PNG")
        return

    plt.figure(figsize=(9, 5))
    if train_losses:
        plt.plot(train_steps, train_losses, marker="o", linewidth=1.5, label="train loss")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, marker="o", linewidth=1.5, label="eval loss")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("K2 Vision Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "training_loss.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"saved training history to {history_path}")
    print(f"saved loss plot to {plot_path}")


def run_pipeline(
    image_path,
    k2_dir=K2_MODEL_DIR,
    trained_dir=K2_FINAL_DIR,
    projector_path=None,
    vision_adapter_dir=None,
    fault_overlay_head_path=None,
    fault_overlay_output=None,
    fault_overlay_threshold=0.5,
    vision_token_drop_rate=0.0,
    question=None,
):
    image = Image.open(image_path).convert("RGB")
    trained_dir = Path(trained_dir)
    vision_adapter_dir = resolve_existing_path(
        vision_adapter_dir,
        trained_dir / "qwen_vision_adapter",
        K2_TRAINED_VISION_ADAPTER_DIR,
        VISION_ADAPTER_DIR,
    )
    projector_path = resolve_existing_path(
        projector_path,
        trained_dir / "k2_qwen_vision_projector.pt",
        K2_TRAINED_PROJECTOR,
        VISION_PREFIX_PROJECTOR,
    )
    fault_overlay_head_path = resolve_existing_path(
        fault_overlay_head_path,
        trained_dir / "fault_overlay_head.pt",
    )
    qwen_model, qwen_processor = load_qwen_vision_encoder(adapter_dir=vision_adapter_dir)
    k2_lora_dir = trained_dir / "k2_lora_adapter"
    k2_model, k2_tokenizer = load_k2(
        k2_dir,
        lora_adapter_dir=k2_lora_dir if k2_lora_dir.exists() else None,
    )

    vision_hidden_size = get_hidden_size(qwen_model)
    k2_hidden_size = get_hidden_size(k2_model)
    projector = load_or_create_projector(
        vision_hidden_size,
        k2_hidden_size,
        projector_path,
    )
    projector = projector.to(k2_model.get_input_embeddings().weight.device)
    projector.eval()
    fault_overlay_head = None
    if fault_overlay_head_path is not None and fault_overlay_head_path.exists():
        fault_overlay_head = load_or_create_fault_overlay_head(vision_hidden_size, fault_overlay_head_path)
        fault_overlay_head = fault_overlay_head.to(k2_model.get_input_embeddings().weight.device)
        fault_overlay_head.eval()

    if fault_overlay_head is None:
        qwen_vision_latent = encode_image_with_qwen(
            qwen_model,
            qwen_processor,
            image,
            token_drop_rate=vision_token_drop_rate,
        )
        fault_outputs = {}
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": IMAGE_PROMPT},
                ],
            }
        ]
        qwen_prompt = qwen_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        qwen_inputs = qwen_processor(text=[qwen_prompt], images=[[image]], return_tensors="pt")
        qwen_inputs = {
            key: value.to(next(qwen_model.parameters()).device) if hasattr(value, "to") else value
            for key, value in qwen_inputs.items()
        }
        with torch.no_grad():
            qwen_vision_latent, vision_tokens, image_mask = encode_qwen_inputs(
                qwen_model,
                qwen_processor,
                qwen_inputs,
                token_drop_rate=vision_token_drop_rate,
                return_tokens=True,
            )
            head_device = next(fault_overlay_head.parameters()).device
            fault_outputs = fault_overlay_head(
                vision_tokens.to(head_device),
                image_mask.to(head_device),
                qwen_vision_latent.to(head_device),
            )
    qwen_vision_latent = qwen_vision_latent.to(
        device=next(projector.parameters()).device,
        dtype=next(projector.parameters()).dtype,
    )
    with torch.no_grad():
        visual_prefix = projector(qwen_vision_latent)

    prompt = build_k2_vqa_prompt(question) if question else K2_PROMPT
    answer = generate_with_visual_prefix(k2_model, k2_tokenizer, visual_prefix, prompt)

    print("k2 attached vision pipeline")
    print(f"image: {image_path}")
    if question:
        print(f"question: {question}")
    print(f"k2_answer: {answer}")
    if fault_outputs:
        presence = torch.sigmoid(fault_outputs["fault_presence_logits"]).detach().cpu().flatten()[0].item()
        overlay = torch.sigmoid(fault_outputs["fault_overlay_logits"]).detach().cpu()
        print(f"fault_presence: {presence:.4f}")
        print(f"fault_overlay_mean: {float(overlay.mean()):.4f}")
        if fault_overlay_output is not None:
            output_path = save_fault_overlay_prediction(
                image,
                overlay[0],
                fault_overlay_output,
                threshold=fault_overlay_threshold,
            )
            print(f"fault_overlay_output: {output_path}")


def train_attached_vision(
    k2_dir=K2_MODEL_DIR,
    output_dir=K2_VISION_OUTPUT_DIR,
    projector_path=None,
    vision_token_drop_rate=VISION_TOKEN_DROP_RATE,
    train_jsonl=DEFAULT_TRAIN_JSONL,
    eval_jsonl=DEFAULT_EVAL_JSONL,
    epochs=1,
    logging_steps=10,
    k2_max_length=2048,
    do_eval=False,
    train_k2_lora=True,
    k2_lora_r=8,
    train_fault_overlay_head=True,
    allow_pseudo_fault_labels=False,
    train_qwen_vision_adapter=True,
    max_image_side=DEFAULT_MAX_IMAGE_SIDE,
):
    output_dir = Path(output_dir)
    final_dir = final_dir_for(output_dir)
    vision_adapter_dir = resolve_existing_path(
        adapter_dir_for(output_dir),
        K2_TRAINED_VISION_ADAPTER_DIR,
        VISION_ADAPTER_DIR,
    )
    projector_path = resolve_existing_path(
        projector_path,
        projector_path_for(output_dir),
        K2_TRAINED_PROJECTOR,
        VISION_PREFIX_PROJECTOR,
    )
    qwen_model, qwen_processor = load_qwen_vision_encoder(
        trainable=train_qwen_vision_adapter,
        adapter_dir=vision_adapter_dir,
    )
    k2_lora_dir = resolve_existing_path(
        k2_lora_dir_for(output_dir),
        K2_TRAINED_LORA_DIR,
    )
    k2_model, k2_tokenizer = load_k2(
        k2_dir,
        lora_adapter_dir=k2_lora_dir if train_k2_lora and k2_lora_dir.exists() else None,
        is_trainable=train_k2_lora,
    )
    if train_k2_lora:
        k2_model = add_k2_lora(k2_model, r=k2_lora_r)
    if hasattr(k2_model.config, "use_cache"):
        k2_model.config.use_cache = False

    vision_hidden_size = get_hidden_size(qwen_model)
    k2_hidden_size = get_hidden_size(k2_model)
    projector = load_or_create_projector(
        vision_hidden_size,
        k2_hidden_size,
        projector_path,
    )
    projector = projector.to(k2_model.get_input_embeddings().weight.device)
    projector.train()
    fault_overlay_head = None
    if train_fault_overlay_head:
        fault_overlay_head = load_or_create_fault_overlay_head(
            vision_hidden_size,
            fault_overlay_head_path_for(output_dir),
        )
        fault_overlay_head = fault_overlay_head.to(k2_model.get_input_embeddings().weight.device)
        fault_overlay_head.train()

    model = FrozenK2VisionModel(
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
        k2_model=k2_model,
        projector=projector,
        fault_overlay_head=fault_overlay_head,
        vision_token_drop_rate=vision_token_drop_rate,
    )

    train_dataset = K2VisionDataset(
        train_jsonl,
        allow_pseudo_fault_labels=allow_pseudo_fault_labels,
        max_image_side=max_image_side,
    )
    eval_dataset = (
        K2VisionDataset(
            eval_jsonl,
            allow_pseudo_fault_labels=allow_pseudo_fault_labels,
            max_image_side=max_image_side,
        )
        if do_eval
        else None
    )
    print(f"max image side for Qwen training inputs: {max_image_side}")
    print(f"train fault label sources: {train_dataset.fault_label_source_counts}")
    if train_fault_overlay_head and train_dataset.fault_label_source_counts["real"] == 0:
        if train_dataset.fault_label_source_counts["pseudo"] > 0:
            print("warning: fault overlay head will train only from pseudo-labels because no real mask labels were found.")
        else:
            print("warning: no fault overlay labels found; auxiliary fault head receives no supervised loss.")
    if eval_dataset is not None:
        print(f"eval fault label sources: {eval_dataset.fault_label_source_counts}")
    args = TrainingArguments(
        output_dir=Path(output_dir).as_posix(),
        logging_dir="logs",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        weight_decay=0.02,
        warmup_steps=25,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="epoch" if do_eval else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )

    trainer = K2VisionTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=K2VisionCollator(qwen_processor, k2_tokenizer, max_length=k2_max_length),
    )
    trainer.train(resume_from_checkpoint=latest_checkpoint(output_dir))
    if do_eval:
        trainer.evaluate()
    save_training_history(trainer.state.log_history, output_dir)
    trainer.save_model(final_dir.as_posix())
    qwen_processor.save_pretrained(final_dir / "qwen_vision_adapter")
    k2_tokenizer.save_pretrained(final_dir / "k2_tokenizer")
    torch.save(projector.state_dict(), final_dir / "k2_qwen_vision_projector.pt")
    if fault_overlay_head is not None:
        torch.save(fault_overlay_head.state_dict(), final_dir / "fault_overlay_head.pt")
    if isinstance(k2_model, PeftModel):
        k2_model.save_pretrained(final_dir / "k2_lora_adapter")
    Path(projector_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), projector_path)
    print(f"saved trained K2 vision adapter to {final_dir / 'qwen_vision_adapter'}")
    print(f"saved trained projector to {final_dir / 'k2_qwen_vision_projector.pt'}")
    if fault_overlay_head is not None:
        print(f"saved trained fault overlay head to {final_dir / 'fault_overlay_head.pt'}")
    if isinstance(k2_model, PeftModel):
        print(f"saved trained K2 LoRA adapter to {final_dir / 'k2_lora_adapter'}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("image", help="Path to seismic image.")
    run_parser.add_argument(
        "--k2-dir",
        default=K2_MODEL_DIR.as_posix(),
        help="Local folder where the full K2 repository is/will be downloaded.",
    )
    run_parser.add_argument(
        "--trained-dir",
        default=K2_FINAL_DIR.as_posix(),
        help="Folder containing saved qwen_vision_adapter and k2_qwen_vision_projector.pt.",
    )
    run_parser.add_argument(
        "--vision-adapter",
        default=None,
        help="Optional explicit Qwen vision adapter folder. Defaults to --trained-dir/qwen_vision_adapter.",
    )
    run_parser.add_argument(
        "--projector",
        default=None,
        help="Optional explicit Qwen-vision-to-K2-prefix projector path. Defaults to --trained-dir/k2_qwen_vision_projector.pt.",
    )
    run_parser.add_argument(
        "--fault-overlay-head",
        default=None,
        help="Optional trained fault overlay head path. Defaults to --trained-dir/fault_overlay_head.pt when present.",
    )
    run_parser.add_argument(
        "--fault-overlay-output",
        default=None,
        help="Optional PNG path for the predicted model fault overlay.",
    )
    run_parser.add_argument(
        "--fault-overlay-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to emphasize predicted fault overlay pixels.",
    )
    run_parser.add_argument(
        "--vision-token-drop-rate",
        type=float,
        default=0.0,
        help="Image-token drop rate during inference. Default keeps all image tokens.",
    )
    run_parser.add_argument(
        "--question",
        default=None,
        help="Optional VQA question. If omitted, uses the default seismic classification prompt.",
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--k2-dir",
        default=K2_MODEL_DIR.as_posix(),
        help="Local folder where the full K2 repository is/will be downloaded.",
    )
    train_parser.add_argument(
        "--output-dir",
        default=K2_VISION_OUTPUT_DIR.as_posix(),
        help="Output folder for trained Qwen vision adapter and projector.",
    )
    train_parser.add_argument(
        "--projector",
        default=None,
        help="Optional extra path to save/load the Qwen-vision-to-K2-prefix projector. A copy is always saved under --output-dir/final.",
    )
    train_parser.add_argument(
        "--vision-token-drop-rate",
        type=float,
        default=VISION_TOKEN_DROP_RATE,
        help="Image-token drop rate during training. Default drops 75 percent.",
    )
    train_parser.add_argument(
        "--max-image-side",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIDE,
        help="Resize training images so the longest side is at most this many pixels before Qwen processing. Use 0 to disable.",
    )
    train_parser.add_argument(
        "--train-jsonl",
        default=DEFAULT_TRAIN_JSONL.as_posix(),
        help="Generated SFT JSONL for training, usually outputs/generated_unicamp_instructions/train_sft.jsonl.",
    )
    train_parser.add_argument(
        "--eval-jsonl",
        default=DEFAULT_EVAL_JSONL.as_posix(),
        help="Generated SFT JSONL for eval, usually outputs/generated_unicamp_instructions/validation_sft.jsonl.",
    )
    train_parser.add_argument(
        "--epochs",
        type=float,
        default=1,
        help="Number of training epochs.",
    )
    train_parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="How often to log training loss.",
    )
    train_parser.add_argument(
        "--k2-max-length",
        type=int,
        default=2048,
        help="Maximum K2 text tokens. Answer tokens are preserved; prompt tokens are truncated first.",
    )
    train_parser.add_argument(
        "--do-eval",
        action="store_true",
        help="Run evaluation each epoch. Disabled by default to avoid OOM on 24GB GPUs.",
    )
    train_parser.add_argument(
        "--no-k2-lora",
        action="store_true",
        help="Disable small K2 decoder LoRA training.",
    )
    train_parser.add_argument(
        "--k2-lora-r",
        type=int,
        default=8,
        help="Rank for the small K2 decoder LoRA.",
    )
    train_parser.add_argument(
        "--no-fault-overlay-head",
        action="store_true",
        help="Disable the auxiliary model-architecture fault overlay head.",
    )
    train_parser.add_argument(
        "--allow-pseudo-fault-labels",
        action="store_true",
        help="Allow FaultNet detections as weak labels when real fault masks/YOLO labels are unavailable.",
    )
    train_parser.add_argument(
        "--freeze-qwen-vision-adapter",
        action="store_true",
        help="Freeze Qwen vision during training. This is the safest 24GB-GPU mode; trains projector, K2 LoRA, and fault head.",
    )

    args = parser.parse_args()
    if args.command == "run":
        run_pipeline(
            Path(args.image),
            Path(args.k2_dir),
            Path(args.trained_dir),
            Path(args.projector) if args.projector else None,
            Path(args.vision_adapter) if args.vision_adapter else None,
            Path(args.fault_overlay_head) if args.fault_overlay_head else None,
            Path(args.fault_overlay_output) if args.fault_overlay_output else None,
            args.fault_overlay_threshold,
            args.vision_token_drop_rate,
            args.question,
        )
    elif args.command == "train":
        train_attached_vision(
            Path(args.k2_dir),
            Path(args.output_dir),
            Path(args.projector) if args.projector else None,
            args.vision_token_drop_rate,
            Path(args.train_jsonl),
            Path(args.eval_jsonl),
            args.epochs,
            args.logging_steps,
            args.k2_max_length,
            args.do_eval,
            not args.no_k2_lora,
            args.k2_lora_r,
            not args.no_fault_overlay_head,
            args.allow_pseudo_fault_labels,
            not args.freeze_qwen_vision_adapter,
            args.max_image_side,
        )


if __name__ == "__main__":
    main()
