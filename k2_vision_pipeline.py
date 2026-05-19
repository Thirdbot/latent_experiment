import argparse
import json
from pathlib import Path

import unsloth
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image
from torch import nn
from unsloth import FastVisionModel, is_bfloat16_supported
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


K2_REPO_ID = "daven3/k2"
K2_MODEL_DIR = Path("models/k2")
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
NUM_VISION_PREFIX_TOKENS = 8
VISION_TOKEN_DROP_RATE = 0.75

IMAGE_PROMPT = (
    "You are encoding a seismic reflection image for a geoscience language model. "
    "Focus on reflector continuity, amplitude, geometry, offset, chaos, channels, salt, "
    "and transparency."
)

K2_PROMPT = (
    "The preceding soft visual prefix comes from a Qwen-VL image encoder that processed "
    "a seismic reflection image. Interpret the image. Classify it using exactly one of: "
    "Boring, Bright_Planar, Bright_Chaotic, Channel, Converging_Amplitudes, Fault, Salt, "
    "Transparent_Planar. Answer in exactly two concise sentences. Sentence 1: <Class>. "
    "Sentence 2: one visible seismic reason using reflector continuity, amplitude, "
    "geometry, offset, chaos, or transparency. Do not use markdown or bullets."
)

def latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    return get_last_checkpoint(output_dir.as_posix())


def final_dir_for(output_dir):
    return Path(output_dir) / "final"


def adapter_dir_for(output_dir):
    return final_dir_for(output_dir) / "qwen_vision_adapter"


def projector_path_for(output_dir):
    return final_dir_for(output_dir) / "k2_qwen_vision_projector.pt"


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


def load_k2(model_dir=K2_MODEL_DIR):
    model_dir = download_k2(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir.as_posix(),
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir.as_posix(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_qwen_vision_encoder(trainable=False, adapter_dir=None):
    model, processor = FastVisionModel.from_pretrained(
        VISION_MODEL_NAME,
        load_in_4bit=True,
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
        for name, param in model.named_parameters():
            is_language_param = "language" in name or "lm_head" in name or "embed_tokens" in name
            can_train = param.is_floating_point() or param.is_complex()
            param.requires_grad = can_train and not is_language_param
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
        outputs = qwen_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states[-1]
    image_mask = inputs["input_ids"].eq(image_token_id)
    image_mask = apply_image_token_dropout(
        image_mask,
        drop_rate=token_drop_rate,
        training=token_drop_rate > 0,
    )
    mask = image_mask.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def encode_qwen_inputs(qwen_model, qwen_processor, inputs, token_drop_rate=VISION_TOKEN_DROP_RATE):
    image_token_id = get_image_token_id(qwen_model, qwen_processor)
    outputs = qwen_model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states[-1]
    image_mask = inputs["input_ids"].eq(image_token_id)
    image_mask = apply_image_token_dropout(
        image_mask,
        drop_rate=token_drop_rate,
        training=qwen_model.training,
    )
    mask = image_mask.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


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

    with torch.no_grad():
        generated_ids = k2_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=96,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=k2_tokenizer.pad_token_id,
            eos_token_id=k2_tokenizer.eos_token_id,
        )
    return k2_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


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


class K2VisionDataset(Dataset):
    def __init__(self, jsonl_path):
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Generated SFT JSONL not found: {jsonl_path}")
        self.records = []
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.records.append(json.loads(line))
        if not self.records:
            raise ValueError(f"No records found in {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = Path(record["image"])
        answer = get_message_text(record.get("messages", [])) or record.get("chosen_answer", "")
        question = get_question(record)
        return {
            "image": Image.open(image_path).convert("RGB"),
            "question": question,
            "answer": answer,
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
            (
                "The preceding soft visual prefix comes from a Qwen-VL image encoder that "
                "processed a 2D seismic image. Answer the user's seismic interpretation "
                f"question using visible evidence only.\n\nQuestion: {example['question']}\nAnswer:"
            )
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
        return batch


class FrozenK2VisionModel(nn.Module):
    def __init__(
        self,
        qwen_model,
        qwen_processor,
        k2_model,
        projector,
        vision_token_drop_rate=VISION_TOKEN_DROP_RATE,
    ):
        super().__init__()
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.k2_model = k2_model
        self.projector = projector
        self.vision_token_drop_rate = vision_token_drop_rate

        self.k2_model.eval()
        for param in self.k2_model.parameters():
            param.requires_grad = False

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
        self.qwen_model.save_pretrained(output_dir / "qwen_vision_adapter")
        torch.save(self.projector.state_dict(), output_dir / "k2_qwen_vision_projector.pt")

    def forward(self, **batch):
        qwen_inputs = {
            key.removeprefix("qwen_"): value.to(next(self.qwen_model.parameters()).device)
            for key, value in batch.items()
            if key.startswith("qwen_")
        }
        vision_latent = encode_qwen_inputs(
            self.qwen_model,
            self.qwen_processor,
            qwen_inputs,
            token_drop_rate=self.vision_token_drop_rate,
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

        return self.k2_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )


class K2VisionTrainer(Trainer):
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
    vision_token_drop_rate=0.0,
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
    qwen_model, qwen_processor = load_qwen_vision_encoder(adapter_dir=vision_adapter_dir)
    k2_model, k2_tokenizer = load_k2(k2_dir)

    vision_hidden_size = get_hidden_size(qwen_model)
    k2_hidden_size = get_hidden_size(k2_model)
    projector = load_or_create_projector(
        vision_hidden_size,
        k2_hidden_size,
        projector_path,
    )
    projector = projector.to(k2_model.get_input_embeddings().weight.device)
    projector.eval()

    qwen_vision_latent = encode_image_with_qwen(
        qwen_model,
        qwen_processor,
        image,
        token_drop_rate=vision_token_drop_rate,
    )
    qwen_vision_latent = qwen_vision_latent.to(
        device=next(projector.parameters()).device,
        dtype=next(projector.parameters()).dtype,
    )
    with torch.no_grad():
        visual_prefix = projector(qwen_vision_latent)

    answer = generate_with_visual_prefix(k2_model, k2_tokenizer, visual_prefix, K2_PROMPT)

    print("k2 attached vision pipeline")
    print(f"image: {image_path}")
    print(f"k2_answer: {answer}")


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
        trainable=True,
        adapter_dir=vision_adapter_dir,
    )
    k2_model, k2_tokenizer = load_k2(k2_dir)
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

    model = FrozenK2VisionModel(
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
        k2_model=k2_model,
        projector=projector,
        vision_token_drop_rate=vision_token_drop_rate,
    )

    train_dataset = K2VisionDataset(train_jsonl)
    eval_dataset = K2VisionDataset(eval_jsonl)
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
        eval_strategy="epoch",
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
    trainer.evaluate()
    save_training_history(trainer.state.log_history, output_dir)
    trainer.save_model(final_dir.as_posix())
    qwen_processor.save_pretrained(final_dir / "qwen_vision_adapter")
    k2_tokenizer.save_pretrained(final_dir / "k2_tokenizer")
    torch.save(projector.state_dict(), final_dir / "k2_qwen_vision_projector.pt")
    Path(projector_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), projector_path)
    print(f"saved trained K2 vision adapter to {final_dir / 'qwen_vision_adapter'}")
    print(f"saved trained projector to {final_dir / 'k2_qwen_vision_projector.pt'}")


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
        "--vision-token-drop-rate",
        type=float,
        default=0.0,
        help="Image-token drop rate during inference. Default keeps all image tokens.",
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

    args = parser.parse_args()
    if args.command == "run":
        run_pipeline(
            Path(args.image),
            Path(args.k2_dir),
            Path(args.trained_dir),
            Path(args.projector) if args.projector else None,
            Path(args.vision_adapter) if args.vision_adapter else None,
            args.vision_token_drop_rate,
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
        )


if __name__ == "__main__":
    main()
