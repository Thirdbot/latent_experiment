import json
import os
from pathlib import Path
import time

import unsloth
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer
from unsloth import FastVisionModel, is_bfloat16_supported

from seismic_k2.config import (
    DEFAULT_EVAL_DATA,
    DEFAULT_TRAIN_DATA,
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
from seismic_k2.vlm.dataset import ExportedMultimodalDataset


IMAGE_PROMPT = "Represent this seismic slice for downstream geoscience question answering."
SEG_TOKEN = "<SEG>"
DEFAULT_MASK_OUTPUT_SIZE = 256
DEFAULT_MASK_LOSS_WEIGHT = 1.0
DEFAULT_MASK_DICE_WEIGHT = 1.0
DEFAULT_WANDB_PROJECT = "k2-seismic-lisa"
K2_PROMPT = (
    "You are a seismic interpretation assistant. Use the visual prefix from the image "
    "to answer concisely with geological reasoning.\nAnswer:"
)
K2_VQA_PROMPT_TEMPLATE = (
    "You are a seismic interpretation assistant. Use the visual prefix from the image "
    "to answer the question.\nQuestion: {question}\nAnswer:"
)
DEFAULT_MAX_IMAGE_SIDE = 512


def final_dir_for(output_dir):
    return Path(output_dir) / "final"


def adapter_dir_for(output_dir):
    return final_dir_for(output_dir) / "qwen_vision_adapter"


def projector_path_for(output_dir):
    return final_dir_for(output_dir) / "k2_qwen_vision_projector.pt"


def k2_lora_dir_for(output_dir):
    return final_dir_for(output_dir) / "k2_lora_adapter"


def segmentation_head_path_for(output_dir):
    return final_dir_for(output_dir) / "segmentation_head.pt"


def resolve_existing_path(*paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    for path in paths:
        if path is not None:
            return Path(path)
    return None


def has_peft_adapter(path):
    path = Path(path)
    return (
        path.exists()
        and (path / "adapter_config.json").exists()
        and ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists())
    )


def resolve_existing_peft_adapter(*paths):
    for path in paths:
        if path is not None and has_peft_adapter(path):
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
    print(f"snapshot_download: {K2_REPO_ID} -> {model_dir}")
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
        for filename in ("tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    )


def load_k2_tokenizer(model_dir=K2_MODEL_DIR, tokenizer_name=K2_TOKENIZER_NAME):
    tokenizer = None
    if has_tokenizer_files(model_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(Path(model_dir).as_posix(), use_fast=False, trust_remote_code=True)
        except Exception:
            tokenizer = None

    if tokenizer is None or getattr(tokenizer, "vocab_size", 0) < 1000:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    if getattr(tokenizer, "vocab_size", 0) < 1000:
        raise ValueError(
            f"Invalid K2 tokenizer vocab_size={getattr(tokenizer, 'vocab_size', None)}. "
            "K2 needs a LLaMA tokenizer, not incomplete local tokenizer files."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_k2(model_dir=K2_MODEL_DIR, tokenizer_name=K2_TOKENIZER_NAME, lora_adapter_dir=None, is_trainable=False):
    model_dir = download_k2(model_dir)
    tokenizer = load_k2_tokenizer(model_dir, tokenizer_name)
    print(f"loaded K2 tokenizer vocab_size={tokenizer.vocab_size}")

    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir.as_posix(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if is_trainable and quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    if lora_adapter_dir is not None and has_peft_adapter(lora_adapter_dir):
        model = PeftModel.from_pretrained(model, Path(lora_adapter_dir).as_posix(), is_trainable=is_trainable)
        print(f"loaded K2 LoRA adapter from {lora_adapter_dir}")
    elif lora_adapter_dir is not None:
        print(f"K2 LoRA adapter not found or incomplete at {lora_adapter_dir}; using base K2")
    model.eval()
    return model, tokenizer


def ensure_seg_token(model, tokenizer):
    token_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN]})
        token_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        resize_target = model.get_base_model() if isinstance(model, PeftModel) else model
        try:
            resize_target.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            resize_target.resize_token_embeddings(len(tokenizer))
        print(f"added {SEG_TOKEN} token with id={token_id}")
    return token_id


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
        adapter_dir = resolve_existing_peft_adapter(K2_TRAINED_VISION_ADAPTER_DIR, VISION_ADAPTER_DIR)
    else:
        adapter_dir = Path(adapter_dir)
    has_vision_adapter = adapter_dir is not None and has_peft_adapter(adapter_dir)
    if has_vision_adapter:
        model = PeftModel.from_pretrained(model, adapter_dir.as_posix(), is_trainable=trainable)
        if trainable:
            print(f"loaded trainable Qwen vision adapter from {adapter_dir}")
        else:
            model = model.merge_and_unload()
            print(f"merged Qwen vision adapter from {adapter_dir}")
    else:
        print(f"vision adapter not found or incomplete at {adapter_dir}; using base Qwen-VL")

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
        for row in empty_rows.nonzero(as_tuple=False).flatten().tolist():
            token_indices = image_mask[row].nonzero(as_tuple=False).flatten()
            selected = token_indices[torch.randint(token_indices.numel(), (1,), device=image_mask.device)]
            keep_mask[row, selected] = True
    return keep_mask


class VisionPrefixProjector(nn.Module):
    # Custom bridge: Qwen and K2 do not share a native multimodal interface.
    # This MLP turns one pooled Qwen image embedding into a few fake K2 input
    # embeddings that are prepended before the text prompt.
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
        print(f"projector not found at {projector_path}; using randomly initialized projector")
    return projector


def qwen_forward_hidden_states(qwen_model, inputs):
    kwargs = {
        **inputs,
        "output_hidden_states": True,
        "return_dict": True,
        "use_cache": False,
        "logits_to_keep": 1,
    }
    try:
        return qwen_model(**kwargs)
    except TypeError:
        kwargs.pop("logits_to_keep", None)
        return qwen_model(**kwargs)


def encode_qwen_inputs(qwen_model, qwen_processor, inputs, token_drop_rate=VISION_TOKEN_DROP_RATE, return_tokens=False):
    image_token_id = get_image_token_id(qwen_model, qwen_processor)
    outputs = qwen_forward_hidden_states(qwen_model, inputs)
    hidden_states = outputs.hidden_states[-1]
    image_mask = inputs["input_ids"].eq(image_token_id)
    image_mask = apply_image_token_dropout(image_mask, drop_rate=token_drop_rate, training=qwen_model.training)
    mask = image_mask.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    vision_latent = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if return_tokens:
        return vision_latent, hidden_states, image_mask
    return vision_latent


def encode_image_with_qwen(qwen_model, qwen_processor, image, token_drop_rate=0.0, return_tokens=False):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": IMAGE_PROMPT}]}]
    prompt = qwen_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = qwen_processor(text=[prompt], images=[[image]], return_tensors="pt")
    device = next(qwen_model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        return encode_qwen_inputs(qwen_model, qwen_processor, inputs, token_drop_rate=token_drop_rate, return_tokens=return_tokens)


def generate_with_visual_prefix(k2_model, k2_tokenizer, visual_prefix, prompt, max_new_tokens=128):
    device = k2_model.get_input_embeddings().weight.device
    text_inputs = k2_tokenizer(prompt, return_tensors="pt")
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    text_embeddings = k2_model.get_input_embeddings()(text_inputs["input_ids"])
    visual_prefix = visual_prefix.to(device=device, dtype=text_embeddings.dtype)
    inputs_embeds = torch.cat([visual_prefix, text_embeddings], dim=1)
    prefix_mask = torch.ones(visual_prefix.size()[:2], dtype=text_inputs["attention_mask"].dtype, device=device)
    attention_mask = torch.cat([prefix_mask, text_inputs["attention_mask"]], dim=1)

    generation_kwargs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
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
    text = k2_tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if "Answer:" in text:
        text = text.rsplit("Answer:", 1)[-1].strip()
    return text or f"[empty decoded output] token_ids={token_ids[:64]}"


def get_k2_hidden_for_seg_token(k2_model, k2_tokenizer, visual_prefix, prompt, answer, seg_token_id):
    device = k2_model.get_input_embeddings().weight.device
    text = f"{prompt}{answer}\n{SEG_TOKEN}"
    text_inputs = k2_tokenizer(text, return_tensors="pt", add_special_tokens=True)
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    input_ids = text_inputs["input_ids"]
    seg_positions = input_ids.eq(seg_token_id).nonzero(as_tuple=False)
    if seg_positions.numel() == 0:
        raise ValueError(f"{SEG_TOKEN} was not found in forced segmentation prompt.")

    text_embeddings = k2_model.get_input_embeddings()(input_ids)
    visual_prefix = visual_prefix.to(device=device, dtype=text_embeddings.dtype)
    inputs_embeds = torch.cat([visual_prefix, text_embeddings], dim=1)
    prefix_mask = torch.ones(visual_prefix.size()[:2], dtype=text_inputs["attention_mask"].dtype, device=device)
    attention_mask = torch.cat([prefix_mask, text_inputs["attention_mask"]], dim=1)

    with torch.no_grad():
        outputs = k2_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
    seg_position = int(seg_positions[-1, 1].item()) + visual_prefix.size(1)
    return outputs.hidden_states[-1][:, seg_position]


def build_k2_vqa_prompt(question):
    return K2_VQA_PROMPT_TEMPLATE.format(question=question)


class K2VisionCollator:
    def __init__(self, qwen_processor, k2_tokenizer, max_length=2048):
        self.qwen_processor = qwen_processor
        self.k2_tokenizer = k2_tokenizer
        self.max_length = max_length
        self.seg_token_id = self.k2_tokenizer.convert_tokens_to_ids(SEG_TOKEN)

    def __call__(self, examples):
        qwen_messages = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": IMAGE_PROMPT}]}]
            for _ in examples
        ]
        qwen_prompts = [
            self.qwen_processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            for message in qwen_messages
        ]
        qwen_inputs = self.qwen_processor(
            text=qwen_prompts,
            images=[[example["image"]] for example in examples],
            return_tensors="pt",
            padding=True,
        )

        input_id_rows = []
        label_rows = []
        attention_rows = []
        pad_token_id = self.k2_tokenizer.pad_token_id or self.k2_tokenizer.eos_token_id

        for example in examples:
            prompt_ids = self.k2_tokenizer(
                build_k2_vqa_prompt(example["question"]),
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            answer_text = f"\n{example['answer']}"
            if example.get("mask_valid") is not None and float(example["mask_valid"].item()) > 0.5:
                answer_text = f"{answer_text}\n{SEG_TOKEN}"
            answer_ids = self.k2_tokenizer(answer_text, add_special_tokens=False, truncation=False)[
                "input_ids"
            ]
            if not answer_ids:
                raise ValueError(f"empty answer target for record {example.get('id')}")

            if len(answer_ids) >= self.max_length:
                answer_ids = answer_ids[: self.max_length - 1]
            max_prompt_len = self.max_length - len(answer_ids)
            if max_prompt_len <= 0:
                raise ValueError("no room left for prompt tokens; increase the K2 text length limit")
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

        batch = {f"qwen_{key}": value for key, value in qwen_inputs.items()}
        batch["k2_input_ids"] = torch.tensor(input_id_rows, dtype=torch.long)
        batch["k2_attention_mask"] = torch.tensor(attention_rows, dtype=torch.long)
        batch["labels"] = torch.tensor(label_rows, dtype=torch.long)
        batch["target_masks"] = torch.stack([example["mask"] for example in examples])
        batch["mask_valid"] = torch.stack([example["mask_valid"] for example in examples])
        return batch


class LisaStyleSegmentationHead(nn.Module):
    # Custom LISA-style head: when masks exist, the hidden state at <SEG>
    # becomes a query that scores Qwen image tokens and produces a mask.
    def __init__(self, vision_hidden_size, k2_hidden_size, output_size=DEFAULT_MASK_OUTPUT_SIZE):
        super().__init__()
        self.output_size = output_size
        hidden = max(vision_hidden_size // 2, 128)
        self.seg_projector = nn.Sequential(
            nn.LayerNorm(k2_hidden_size),
            nn.Linear(k2_hidden_size, vision_hidden_size),
            nn.GELU(),
            nn.Linear(vision_hidden_size, vision_hidden_size),
        )
        self.token_projector = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, vision_hidden_size),
        )
        self.bias = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, 1),
        )

    def forward(self, vision_tokens, image_mask, seg_hidden):
        query = self.seg_projector(seg_hidden)
        tokens = self.token_projector(vision_tokens)
        scores = (tokens * query.unsqueeze(1)).sum(dim=-1) / max(query.size(-1) ** 0.5, 1.0)
        scores = scores + self.bias(vision_tokens).squeeze(-1)
        scores = scores.masked_fill(~image_mask, 0.0)

        batch_size = vision_tokens.size(0)
        token_counts = image_mask.sum(dim=1).clamp_min(1)
        max_tokens = int(token_counts.max().item())
        side = max(int(max_tokens**0.5), 1)
        while side * side < max_tokens:
            side += 1

        grid = vision_tokens.new_zeros((batch_size, 1, side * side))
        for row in range(batch_size):
            values = scores[row, image_mask[row]]
            grid[row, 0, : values.numel()] = values
        grid = grid.view(batch_size, 1, side, side)
        return F.interpolate(grid, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)


def load_or_create_segmentation_head(vision_hidden_size, k2_hidden_size, head_path=None, output_size=DEFAULT_MASK_OUTPUT_SIZE):
    head = LisaStyleSegmentationHead(vision_hidden_size, k2_hidden_size, output_size=output_size)
    if head_path is not None and Path(head_path).exists():
        state = torch.load(head_path, map_location="cpu")
        head.load_state_dict(state)
        print(f"loaded segmentation head from {head_path}")
    elif head_path is not None:
        print(f"segmentation head not found at {head_path}; using randomly initialized head")
    return head


def dice_loss_from_logits(logits, target):
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3))
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


class K2VisionModel(nn.Module):
    def __init__(
        self,
        qwen_model,
        qwen_processor,
        k2_model,
        projector,
        segmentation_head=None,
        seg_token_id=None,
        vision_token_drop_rate=VISION_TOKEN_DROP_RATE,
        mask_loss_weight=DEFAULT_MASK_LOSS_WEIGHT,
        mask_dice_weight=DEFAULT_MASK_DICE_WEIGHT,
    ):
        super().__init__()
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.k2_model = k2_model
        self.projector = projector
        self.segmentation_head = segmentation_head
        self.seg_token_id = seg_token_id
        self.vision_token_drop_rate = vision_token_drop_rate
        self.mask_loss_weight = mask_loss_weight
        self.mask_dice_weight = mask_dice_weight
        self.qwen_trainable = any(param.requires_grad for param in self.qwen_model.parameters())
        if not self.qwen_trainable:
            self.qwen_model.eval()
        self.k2_trainable = any(param.requires_grad for param in self.k2_model.parameters())
        self.k2_model.train(self.k2_trainable)
        for name, param in self.k2_model.named_parameters():
            param.requires_grad = param.requires_grad and "lora_" in name
        self.k2_trainable = any(param.requires_grad for param in self.k2_model.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.qwen_trainable:
            self.qwen_model.eval()
        self.k2_model.train(mode and self.k2_trainable)
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.qwen_model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.qwen_model.gradient_checkpointing_enable()
            else:
                self.qwen_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

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
        if self.segmentation_head is not None:
            torch.save(self.segmentation_head.state_dict(), output_dir / "segmentation_head.pt")

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

        # Custom vision-to-language step: project Qwen's pooled image feature
        # into K2 embedding space and prepend it to the text embeddings.
        projector_device = next(self.projector.parameters()).device
        vision_latent = vision_latent.to(device=projector_device, dtype=next(self.projector.parameters()).dtype)
        visual_prefix = self.projector(vision_latent)

        k2_device = self.k2_model.get_input_embeddings().weight.device
        input_ids = batch["k2_input_ids"].to(k2_device)
        attention_mask = batch["k2_attention_mask"].to(k2_device)
        labels = batch["labels"].to(k2_device)
        text_embeddings = self.k2_model.get_input_embeddings()(input_ids)
        visual_prefix = visual_prefix.to(device=k2_device, dtype=text_embeddings.dtype)
        inputs_embeds = torch.cat([visual_prefix, text_embeddings], dim=1)
        prefix_mask = torch.ones(visual_prefix.size()[:2], dtype=attention_mask.dtype, device=k2_device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        prefix_labels = torch.full(visual_prefix.size()[:2], -100, dtype=labels.dtype, device=k2_device)
        labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.k2_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=self.segmentation_head is not None,
        )
        if self.segmentation_head is None or self.seg_token_id is None or "target_masks" not in batch:
            return outputs

        seg_mask = input_ids.eq(self.seg_token_id)
        valid = batch["mask_valid"].to(k2_device).flatten() > 0.5
        seg_valid = seg_mask.any(dim=1) & valid
        if not seg_valid.any():
            return outputs

        # Custom mask step: pull the K2 hidden state at <SEG>, combine it with
        # Qwen image-token features, and add BCE + Dice mask loss.
        prefix_len = visual_prefix.size(1)
        hidden_states = outputs.hidden_states[-1]
        seg_positions = []
        for row in range(input_ids.size(0)):
            positions = seg_mask[row].nonzero(as_tuple=False).flatten()
            seg_positions.append(int(positions[-1].item()) if positions.numel() else 0)
        seg_positions = torch.tensor(seg_positions, dtype=torch.long, device=k2_device) + prefix_len
        row_indices = torch.arange(input_ids.size(0), device=k2_device)
        seg_hidden = hidden_states[row_indices, seg_positions]

        head_device = next(self.segmentation_head.parameters()).device
        head_dtype = next(self.segmentation_head.parameters()).dtype
        mask_logits = self.segmentation_head(
            vision_tokens.to(device=head_device, dtype=head_dtype),
            image_mask.to(head_device),
            seg_hidden.to(device=head_device, dtype=head_dtype),
        )
        target_masks = batch["target_masks"].to(device=head_device, dtype=mask_logits.dtype)
        valid_indices = seg_valid.to(head_device)
        bce = F.binary_cross_entropy_with_logits(mask_logits[valid_indices], target_masks[valid_indices])
        dice = dice_loss_from_logits(mask_logits[valid_indices], target_masks[valid_indices])
        outputs.loss = outputs.loss + self.mask_loss_weight * bce + self.mask_dice_weight * dice
        outputs.mask_loss = bce.detach()
        outputs.mask_dice_loss = dice.detach()
        outputs.mask_supervised = valid_indices.sum().detach()
        outputs.mask_logits = mask_logits
        return outputs


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


def save_mask_prediction(mask_prob, output_path, size=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = mask_prob.detach().float().cpu().squeeze().clamp(0, 1).numpy()
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    if size is not None:
        mask_image = mask_image.resize(size, Image.Resampling.BILINEAR)
    mask_image.save(output_path)
    return output_path


def save_overlay_prediction(image, mask_prob, output_path, threshold=0.5, alpha=0.45):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGB")
    mask = mask_prob.detach().float().cpu().squeeze().clamp(0, 1).numpy()
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(base.size, Image.Resampling.BILINEAR)
    mask_array = np.asarray(mask_image).astype(np.float32) / 255.0
    base_array = np.asarray(base).astype(np.float32)

    heat = np.zeros_like(base_array)
    heat[..., 0] = 255.0
    heat[..., 1] = 40.0
    heat[..., 2] = 40.0
    blend = (mask_array * alpha)[..., None]
    overlay = base_array * (1.0 - blend) + heat * blend
    hard_mask = mask_array >= threshold
    overlay[hard_mask] = overlay[hard_mask] * 0.65 + heat[hard_mask] * 0.35
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)
    return output_path


def evaluate_loss(model, data_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def start_wandb_run(enabled, project, entity, run_name, mode, output_dir):
    if not enabled:
        return None
    os.environ.setdefault("WANDB_CONSOLE", "off")
    os.environ.setdefault("WANDB_SILENT", "true")
    wandb_dir = Path(output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    import wandb

    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        dir=wandb_dir.as_posix(),
        config={"output_dir": Path(output_dir).as_posix()},
    )


def run_pipeline(
    image_path,
    k2_dir=K2_MODEL_DIR,
    trained_dir=K2_FINAL_DIR,
    projector_path=None,
    vision_adapter_dir=None,
    vision_token_drop_rate=0.0,
    question=None,
    max_new_tokens=128,
    segmentation_head_path=None,
    mask_output=None,
    overlay_output=None,
    overlay_threshold=0.5,
    mask_output_size=DEFAULT_MASK_OUTPUT_SIZE,
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
    if vision_adapter_dir is not None and not has_peft_adapter(vision_adapter_dir):
        print(f"ignoring incomplete Qwen vision adapter directory: {vision_adapter_dir}")
        vision_adapter_dir = resolve_existing_peft_adapter(K2_TRAINED_VISION_ADAPTER_DIR, VISION_ADAPTER_DIR)
    qwen_model, qwen_processor = load_qwen_vision_encoder(adapter_dir=vision_adapter_dir)
    k2_lora_dir = trained_dir / "k2_lora_adapter"
    k2_model, k2_tokenizer = load_k2(k2_dir, lora_adapter_dir=k2_lora_dir if has_peft_adapter(k2_lora_dir) else None)
    seg_token_id = ensure_seg_token(k2_model, k2_tokenizer)

    vision_hidden_size = get_hidden_size(qwen_model)
    k2_hidden_size = get_hidden_size(k2_model)
    projector = load_or_create_projector(vision_hidden_size, k2_hidden_size, projector_path)
    projector = projector.to(k2_model.get_input_embeddings().weight.device)
    projector.eval()
    segmentation_head_path = resolve_existing_path(
        segmentation_head_path,
        trained_dir / "segmentation_head.pt",
        segmentation_head_path_for(trained_dir.parent),
    )
    segmentation_head = None
    if segmentation_head_path is not None and Path(segmentation_head_path).exists():
        segmentation_head = load_or_create_segmentation_head(
            vision_hidden_size,
            k2_hidden_size,
            segmentation_head_path,
            output_size=mask_output_size,
        )
        segmentation_head = segmentation_head.to(k2_model.get_input_embeddings().weight.device)
        segmentation_head.eval()

    qwen_encoded = encode_image_with_qwen(
        qwen_model,
        qwen_processor,
        image,
        token_drop_rate=vision_token_drop_rate,
        return_tokens=segmentation_head is not None,
    )
    if segmentation_head is None:
        qwen_vision_latent = qwen_encoded
        vision_tokens = None
        image_mask = None
    else:
        qwen_vision_latent, vision_tokens, image_mask = qwen_encoded
    qwen_vision_latent = qwen_vision_latent.to(
        device=next(projector.parameters()).device,
        dtype=next(projector.parameters()).dtype,
    )
    with torch.no_grad():
        visual_prefix = projector(qwen_vision_latent)

    prompt = build_k2_vqa_prompt(question) if question else K2_PROMPT
    answer = generate_with_visual_prefix(k2_model, k2_tokenizer, visual_prefix, prompt, max_new_tokens=max_new_tokens)

    print("k2 attached vision pipeline")
    print(f"image: {image_path}")
    if question:
        print(f"question: {question}")
    print(f"k2_answer: {answer}")
    if segmentation_head is not None and (mask_output is not None or overlay_output is not None):
        seg_hidden = get_k2_hidden_for_seg_token(k2_model, k2_tokenizer, visual_prefix, prompt, answer, seg_token_id)
        head_device = next(segmentation_head.parameters()).device
        head_dtype = next(segmentation_head.parameters()).dtype
        with torch.no_grad():
            mask_logits = segmentation_head(
                vision_tokens.to(device=head_device, dtype=head_dtype),
                image_mask.to(head_device),
                seg_hidden.to(device=head_device, dtype=head_dtype),
            )
            mask_prob = torch.sigmoid(mask_logits)[0]
        print(f"mask_mean: {float(mask_prob.mean()):.4f}")
        if mask_output is not None:
            output_path = save_mask_prediction(mask_prob, mask_output, size=image.size)
            print(f"mask_output: {output_path}")
        if overlay_output is not None:
            output_path = save_overlay_prediction(image, mask_prob, overlay_output, threshold=overlay_threshold)
            print(f"overlay_output: {output_path}")


def train_attached_vision(
    k2_dir=K2_MODEL_DIR,
    output_dir=K2_VISION_OUTPUT_DIR,
    projector_path=None,
    vision_token_drop_rate=VISION_TOKEN_DROP_RATE,
    train_data=DEFAULT_TRAIN_DATA,
    eval_data=DEFAULT_EVAL_DATA,
    epochs=1,
    logging_steps=10,
    k2_max_length=2048,
    do_eval=False,
    train_k2_lora=True,
    k2_lora_r=8,
    train_qwen_vision_adapter=True,
    max_image_side=DEFAULT_MAX_IMAGE_SIDE,
    mask_output_size=DEFAULT_MASK_OUTPUT_SIZE,
    train_segmentation_head=True,
    mask_loss_weight=DEFAULT_MASK_LOSS_WEIGHT,
    mask_dice_weight=DEFAULT_MASK_DICE_WEIGHT,
    wandb_project=DEFAULT_WANDB_PROJECT,
    wandb_entity=None,
    wandb_run_name=None,
    wandb_mode="online",
    disable_wandb=False,
):
    output_dir = Path(output_dir)
    final_dir = final_dir_for(output_dir)
    if wandb_run_name is None:
        wandb_run_name = f"k2-seismic-lisa-{time.strftime('%Y%m%d-%H%M%S')}"
    vision_adapter_dir = resolve_existing_path(adapter_dir_for(output_dir), K2_TRAINED_VISION_ADAPTER_DIR, VISION_ADAPTER_DIR)
    if vision_adapter_dir is not None and not has_peft_adapter(vision_adapter_dir):
        print(f"ignoring incomplete Qwen vision adapter directory: {vision_adapter_dir}")
        vision_adapter_dir = resolve_existing_peft_adapter(K2_TRAINED_VISION_ADAPTER_DIR, VISION_ADAPTER_DIR)
    projector_path = resolve_existing_path(projector_path, projector_path_for(output_dir), K2_TRAINED_PROJECTOR, VISION_PREFIX_PROJECTOR)

    qwen_model, qwen_processor = load_qwen_vision_encoder(trainable=train_qwen_vision_adapter, adapter_dir=vision_adapter_dir)
    k2_lora_dir = resolve_existing_path(k2_lora_dir_for(output_dir), K2_TRAINED_LORA_DIR)
    if k2_lora_dir is not None and not has_peft_adapter(k2_lora_dir):
        print(f"ignoring incomplete K2 LoRA adapter directory: {k2_lora_dir}")
        k2_lora_dir = resolve_existing_peft_adapter(K2_TRAINED_LORA_DIR)
    k2_model, k2_tokenizer = load_k2(
        k2_dir,
        lora_adapter_dir=k2_lora_dir if train_k2_lora and k2_lora_dir is not None else None,
        is_trainable=train_k2_lora,
    )
    if train_k2_lora:
        k2_model = add_k2_lora(k2_model, r=k2_lora_r)
    if hasattr(k2_model.config, "use_cache"):
        k2_model.config.use_cache = False

    vision_hidden_size = get_hidden_size(qwen_model)
    k2_hidden_size = get_hidden_size(k2_model)
    projector = load_or_create_projector(vision_hidden_size, k2_hidden_size, projector_path)
    projector = projector.to(k2_model.get_input_embeddings().weight.device)
    projector.train()

    train_dataset = ExportedMultimodalDataset(train_data, max_image_side=max_image_side, mask_output_size=mask_output_size)
    eval_dataset = (
        ExportedMultimodalDataset(eval_data, max_image_side=max_image_side, mask_output_size=mask_output_size)
        if do_eval
        else None
    )
    train_mask_count = sum(
        1 for record in train_dataset.records if record.get("mask_image") and Path(record["mask_image"]).exists()
    )
    if train_segmentation_head and train_mask_count == 0:
        print("segmentation head disabled: no mask_image files found in train data.")
        train_segmentation_head = False
    seg_token_id = ensure_seg_token(k2_model, k2_tokenizer) if train_segmentation_head else None

    segmentation_head = None
    if train_segmentation_head:
        segmentation_head = load_or_create_segmentation_head(
            vision_hidden_size,
            k2_hidden_size,
            segmentation_head_path_for(output_dir),
            output_size=mask_output_size,
        )
        segmentation_head = segmentation_head.to(k2_model.get_input_embeddings().weight.device)
        segmentation_head.train()

    model = K2VisionModel(
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
        k2_model=k2_model,
        projector=projector,
        segmentation_head=segmentation_head,
        seg_token_id=seg_token_id,
        vision_token_drop_rate=vision_token_drop_rate,
        mask_loss_weight=mask_loss_weight,
        mask_dice_weight=mask_dice_weight,
    )

    print(f"train records: {len(train_dataset)} from {train_data}")
    print(f"train mask records: {train_mask_count}")
    if eval_dataset is not None:
        print(f"eval records: {len(eval_dataset)} from {eval_data}")
    print(f"max image side for Qwen training inputs: {max_image_side}")
    if segmentation_head is not None:
        print(
            "segmentation head enabled: "
            f"seg_token={SEG_TOKEN}, mask_output_size={mask_output_size}, "
            f"bce_weight={mask_loss_weight}, dice_weight={mask_dice_weight}"
        )

    collator = K2VisionCollator(qwen_processor, k2_tokenizer, max_length=k2_max_length)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collator) if eval_dataset else None

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.02)
    wandb_run = start_wandb_run(
        enabled=not disable_wandb,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        mode=wandb_mode,
        output_dir=output_dir,
    )

    log_history = []
    global_step = 0
    gradient_accumulation_steps = 4
    model.train()

    # Standard manual training loop: the only trainable pieces are K2 LoRA,
    # the visual-prefix projector, and the optional mask decoder.
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                metrics = {"step": global_step, "epoch": epoch + 1, "loss": float(outputs.loss.detach().cpu())}
                for name in ("mask_loss", "mask_dice_loss", "mask_supervised"):
                    value = getattr(outputs, name, None)
                    if value is not None:
                        metrics[name] = float(value.detach().float().cpu())
                log_history.append(metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)
                if global_step % logging_steps == 0:
                    print(json.dumps(metrics))

        if eval_loader is not None:
            eval_loss = evaluate_loss(model, eval_loader)
            metrics = {"step": global_step, "epoch": epoch + 1, "eval_loss": eval_loss}
            log_history.append(metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=global_step)
            print(json.dumps(metrics))

    if wandb_run is not None:
        wandb_run.finish()

    save_training_history(log_history, output_dir)
    model.save_pretrained(final_dir.as_posix())
    qwen_processor.save_pretrained(final_dir / "qwen_vision_adapter")
    k2_tokenizer.save_pretrained(final_dir / "k2_tokenizer")
    torch.save(projector.state_dict(), final_dir / "k2_qwen_vision_projector.pt")
    if segmentation_head is not None:
        torch.save(segmentation_head.state_dict(), final_dir / "segmentation_head.pt")
    if isinstance(k2_model, PeftModel):
        k2_model.save_pretrained(final_dir / "k2_lora_adapter")
    Path(projector_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), projector_path)
    print(f"saved trained K2 vision adapter to {final_dir / 'qwen_vision_adapter'}")
    print(f"saved trained projector to {final_dir / 'k2_qwen_vision_projector.pt'}")
    if segmentation_head is not None:
        print(f"saved trained segmentation head to {final_dir / 'segmentation_head.pt'}")
    if isinstance(k2_model, PeftModel):
        print(f"saved trained K2 LoRA adapter to {final_dir / 'k2_lora_adapter'}")

if __name__ == "__main__":
    print("Use main.py for training, scripts/download_k2.py for model download, or scripts/split_dataset.py for data splits.")
