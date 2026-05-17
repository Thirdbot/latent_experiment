from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from pytorch_metric_learning.losses import NTXentLoss, NormalizedSoftmaxLoss, ProxyAnchorLoss
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from unsloth import FastVisionModel, UnslothVisionDataCollator, is_bfloat16_supported


MODEL_NAME = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"

INSTRUCTION = (
    "You are an expert geophysicist interpreting seismic reflection images. "
    "Classify the image using exactly one of: Boring, Bright_Planar, Bright_Chaotic, "
    "Channel, Converging_Amplitudes, Fault, Salt, Transparent_Planar. "
    "Answer in exactly two concise sentences. "
    "Sentence 1: <Class>. "
    "Sentence 2: Give one specific visible reason using seismic terms such as reflector "
    "continuity, amplitude, geometry, offset, chaos, or transparency. "
    "Do not use markdown, bullets, numbering, uncertainty language, or extra explanation."
)

OUTPUT_DIR = Path("outputs/vision_llm_trained")
CUSTOM_OUTPUT_DIR = Path("outputs/vision_llm_trained_custom")

LABEL_MAP = {
    0: "Boring",
    1: "Bright_Planar",
    2: "Bright_Chaotic",
    3: "Channel",
    4: "Converging_Amplitudes",
    5: "Fault",
    6: "Salt",
    7: "Transparent_Planar",
}


def latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    return get_last_checkpoint(output_dir.as_posix())


class SeismicConversationDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("thinkonward/reflection-connection", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        label = LABEL_MAP[int(sample["label"])]
        image = sample["image"].convert("RGB")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": INSTRUCTION},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label},
                    ],
                },
            ],
            "images": [image],
        }


class SeismicImageDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("thinkonward/reflection-connection", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return {
            "image": sample["image"].convert("RGB"),
            "label": int(sample["label"]),
        }


def load_unsloth_vision_model(model_name=MODEL_NAME):
    return FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )


def print_one_eval_example(model, processor, split="test", index=0):
    dataset = load_dataset("thinkonward/reflection-connection", split=split)
    sample = dataset[index]
    label = LABEL_MAP[int(sample["label"])]
    image = sample["image"].convert("RGB")
    image_path = sample.get("image_path") or sample.get("path") or f"{split}[{index}]"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=[prompt],
        images=[[image]],
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    generated_text = processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
    )[0].strip()

    print("one-example evaluation")
    print(f"source: {image_path}")
    print(f"label: {label}")
    print(f"generated: {generated_text}")


def get_hidden_size(model):
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None) is not None:
        return text_config.hidden_size
    if config is not None and getattr(config, "hidden_size", None) is not None:
        return config.hidden_size
    raise ValueError("Could not infer hidden_size from model config.")


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


def coverage_loss(embeddings):
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())

    embeddings = F.normalize(embeddings.float(), dim=-1)
    similarity = embeddings @ embeddings.T
    off_diagonal = similarity[
        ~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    ]
    return off_diagonal.pow(2).mean()


def token_level_alignment_loss(token_latents, vision_latent, attention_mask=None):
    token_latents = F.normalize(token_latents.float(), dim=-1)
    vision_latent = F.normalize(vision_latent.float(), dim=-1)
    distances = 1.0 - F.cosine_similarity(token_latents, vision_latent.unsqueeze(1), dim=-1)

    if attention_mask is None:
        return distances.mean()

    mask = attention_mask.to(distances.device, dtype=distances.dtype)
    return (distances * mask).sum() / mask.sum().clamp_min(1.0)


def anchor_usage_loss(image_probs, decoder_probs):
    probs = torch.cat([image_probs, decoder_probs], dim=0)
    mean_probs = probs.mean(dim=0)
    target = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
    return F.kl_div(mean_probs.clamp_min(1e-8).log(), target, reduction="batchmean")


class DecoderLatentCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": INSTRUCTION},
                    ],
                }
            ]
            for _ in examples
        ]
        prompts = [
            self.processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
            )
            for message in messages
        ]
        inputs = self.processor(
            text=prompts,
            images=[[example["image"]] for example in examples],
            return_tensors="pt",
            padding=True,
        )
        inputs["metric_labels"] = torch.tensor(
            [example["label"] for example in examples],
            dtype=torch.long,
        )
        return inputs


class DecoderLatentAlignmentModel(nn.Module):
    def __init__(
        self,
        model_name=MODEL_NAME,
        vision_adapter_path=OUTPUT_DIR / "final",
        latent_dim=512,
        num_anchors=64,
        anchor_temperature=0.1,
    ):
        super().__init__()

        self.model, self.processor = load_unsloth_vision_model(model_name)
        vision_adapter_path = Path(vision_adapter_path)
        if vision_adapter_path.exists():
            print(f"loaded frozen vision adapter from {vision_adapter_path}")
            vision_model = PeftModel.from_pretrained(
                self.model,
                vision_adapter_path.as_posix(),
                is_trainable=False,
            )
            self.model = vision_model.merge_and_unload()
            print("merged vision adapter into base model before adding decoder adapter")
        else:
            print(f"vision adapter not found at {vision_adapter_path}; using frozen base vision tower")

        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
            target_modules="all-linear",
        )

        for name, param in self.model.named_parameters():
            if "visual" in name or "vision" in name or "merger" in name:
                param.requires_grad = False
            if "lm_head" in name or "embed_tokens" in name:
                param.requires_grad = False

        hidden_size = get_hidden_size(self.model)
        self.image_token_id = get_image_token_id(self.model, self.processor)
        self.decoder_projector = nn.Linear(hidden_size, latent_dim)
        self.vision_projector = nn.Linear(hidden_size, latent_dim)
        self.anchor_temperature = anchor_temperature
        self.anchors = nn.Parameter(torch.randn(num_anchors, latent_dim) * 0.02)

        for param in self.vision_projector.parameters():
            param.requires_grad = False
        for param in self.decoder_projector.parameters():
            param.requires_grad = True

        self.infonce_loss = NTXentLoss(temperature=0.07)
        self.proxy_anchor_loss = ProxyAnchorLoss(
            num_classes=len(LABEL_MAP),
            embedding_size=latent_dim,
        )
        self.metric_softmax_loss = NormalizedSoftmaxLoss(
            num_classes=len(LABEL_MAP),
            embedding_size=latent_dim,
        )

        self.model.print_trainable_parameters()

    def quantize_with_anchors(self, latents):
        anchors = F.normalize(self.anchors.float(), dim=-1)
        latents = F.normalize(latents.float(), dim=-1)
        logits = latents @ anchors.T / self.anchor_temperature
        probs = F.softmax(logits, dim=-1)
        quantized = probs @ anchors
        quantized = F.normalize(quantized.float(), dim=-1)
        return quantized, probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.model.gradient_checkpointing_enable()
            else:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        torch.save(self.state_dict(), output_dir / "pytorch_model.bin")

    def pool_by_mask(self, hidden_states, mask):
        mask = mask.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, metric_labels=None, **inputs):
        attention_mask = inputs.get("attention_mask")

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        image_mask = inputs["input_ids"].eq(self.image_token_id)
        decoder_mask = attention_mask.bool() & ~image_mask if attention_mask is not None else ~image_mask

        with torch.no_grad():
            vision_hidden = self.pool_by_mask(hidden_states.detach(), image_mask)
            vision_hidden = vision_hidden.to(dtype=self.vision_projector.weight.dtype)
            vision_latent = self.vision_projector(vision_hidden)

        decoder_hidden = self.pool_by_mask(hidden_states, decoder_mask)
        decoder_hidden = decoder_hidden.to(dtype=self.decoder_projector.weight.dtype)
        decoder_latent = self.decoder_projector(decoder_hidden)
        token_hidden = hidden_states.to(dtype=self.decoder_projector.weight.dtype)
        token_latents = self.decoder_projector(token_hidden)

        decoder_latent = F.normalize(decoder_latent.float(), dim=-1)
        vision_latent = F.normalize(vision_latent.float(), dim=-1).detach()
        image_anchor_latent, image_anchor_probs = self.quantize_with_anchors(vision_latent)
        decoder_anchor_latent, decoder_anchor_probs = self.quantize_with_anchors(decoder_latent)

        pair_embeddings = torch.cat([decoder_latent, vision_latent], dim=0)
        pair_labels = torch.arange(decoder_latent.size(0), device=decoder_latent.device).repeat(2)

        infonce = self.infonce_loss(pair_embeddings, pair_labels)
        proxy_anchor = self.proxy_anchor_loss(decoder_latent, metric_labels)
        metric_cross_entropy = self.metric_softmax_loss(decoder_latent, metric_labels)
        coverage = coverage_loss(decoder_latent)
        cosine = 1.0 - F.cosine_similarity(decoder_latent, vision_latent, dim=-1).mean()
        token_alignment = token_level_alignment_loss(token_latents, vision_latent, attention_mask)
        anchor_distribution = 0.5 * (
            F.kl_div(
                decoder_anchor_probs.clamp_min(1e-8).log(),
                image_anchor_probs.detach(),
                reduction="batchmean",
            )
            + F.kl_div(
                image_anchor_probs.clamp_min(1e-8).log(),
                decoder_anchor_probs.detach(),
                reduction="batchmean",
            )
        )
        quantized_anchor = 1.0 - F.cosine_similarity(
            decoder_anchor_latent,
            image_anchor_latent.detach(),
            dim=-1,
        ).mean()
        anchor_commitment = (
            F.mse_loss(decoder_latent, decoder_anchor_latent.detach())
            + F.mse_loss(vision_latent, image_anchor_latent.detach())
        )
        usage = anchor_usage_loss(image_anchor_probs, decoder_anchor_probs)
        loss = (
            infonce
            + proxy_anchor
            + metric_cross_entropy
            + 0.1 * coverage
            + cosine
            + 0.1 * token_alignment
            + 0.05 * anchor_distribution
            + 0.05 * quantized_anchor
            + 0.01 * anchor_commitment
            + 0.01 * usage
        )

        return {
            "loss": loss,
            "infonce_loss": infonce.detach(),
            "proxy_anchor_loss": proxy_anchor.detach(),
            "metric_cross_entropy_loss": metric_cross_entropy.detach(),
            "coverage_loss": coverage.detach(),
            "cosine_loss": cosine.detach(),
            "token_alignment_loss": token_alignment.detach(),
            "anchor_distribution_loss": anchor_distribution.detach(),
            "quantized_anchor_loss": quantized_anchor.detach(),
            "anchor_commitment_loss": anchor_commitment.detach(),
            "anchor_usage_loss": usage.detach(),
        }


class LatentAlignmentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        self.model.save_pretrained(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


def train_decoder_with_label():
    model, processor = load_unsloth_vision_model(MODEL_NAME)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=False,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    train_dataset = SeismicConversationDataset("train")
    test_dataset = SeismicConversationDataset("test")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR.as_posix(),
        logging_dir="logs",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        weight_decay=0.02,
        warmup_steps=50,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=args,
        data_collator=UnslothVisionDataCollator(model, processor),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001,
            )
        ],
    )
    trainer.train(resume_from_checkpoint=latest_checkpoint(OUTPUT_DIR))
    trainer.evaluate()
    trainer.save_model((OUTPUT_DIR / "final").as_posix())
    processor.save_pretrained((OUTPUT_DIR / "final").as_posix())
    print_one_eval_example(model, processor)


def train_decoder_without_label():
    model = DecoderLatentAlignmentModel(
        model_name=MODEL_NAME,
        vision_adapter_path=OUTPUT_DIR / "final",
    )
    processor = model.processor

    train_dataset = SeismicImageDataset("train")
    eval_dataset = SeismicImageDataset("test")

    args = TrainingArguments(
        output_dir=CUSTOM_OUTPUT_DIR.as_posix(),
        logging_dir="logs",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        weight_decay=0.02,
        warmup_steps=25,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        prediction_loss_only=True,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )

    trainer = LatentAlignmentTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DecoderLatentCollator(processor),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001,
            )
        ],
    )

    trainer.train(resume_from_checkpoint=latest_checkpoint(CUSTOM_OUTPUT_DIR))
    trainer.evaluate()
    trainer.save_model((CUSTOM_OUTPUT_DIR / "final").as_posix())
    print_one_eval_example(model.model, processor)


def main():
    train_decoder_with_label()
    train_decoder_without_label()


def test_understanding():
    test_dataset = load_dataset("thinkonward/reflection-connection", split="test")
    model, processor = load_unsloth_vision_model(MODEL_NAME)
    if (OUTPUT_DIR / "final").exists():
        model = PeftModel.from_pretrained(model, (OUTPUT_DIR / "final").as_posix(), is_trainable=False)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    selected_data = test_dataset[10]
    label = LABEL_MAP[int(selected_data["label"])]
    print("test:", label)
    inputs = processor(
        text=prompt,
        images=[selected_data["image"].convert("RGB")],
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        repetition_penalty=1.25,
        no_repeat_ngram_size=3,
    )
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(generated_texts[0])


if __name__ == "__main__":
    main()
    # test_understanding()
