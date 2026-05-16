from pathlib import Path
from datasets import load_dataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, EarlyStoppingCallback, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTConfig, SFTTrainer
from pytorch_metric_learning.losses import NTXentLoss, NormalizedSoftmaxLoss, ProxyAnchorLoss

MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"

INSTRUCTION = "You are an expert geophysicist.Using only these  Classify this seismic reflection image using one of: \
        (Boring, Bright_Planar, Bright_Chaotic, Channel, Converging_Amplitudes, Fault, Salt, Transparent_Planar). \
        Then give 2-3 short visual reasons based only on visible patterns. Do not invent details. \
        Describe accurately what you see in this image."

OUTPUT_DIR = Path("outputs/vision_llm_trained")
CUSTOM_OUTPUT_DIR = Path("outputs/vision_llm_trained_custom")
DOMAIN_WORDS = [
    "reflector",
    "reflectors",
    "amplitude",
    "amplitudes",
    "fault",
    "offset",
    "break",
    "discontinuity",
    "discontinuous",
    "chaotic",
    "planar",
    "channel",
    "converging",
    "transparent",
    "layered",
    "continuous",
    "interrupted",
    "salt",
]

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


class SeismicConversationDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("thinkonward/reflection-connection", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        label = LABEL_MAP[sample["label"]]
        image = sample["image"].convert("RGB")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": INSTRUCTION},
                        {"type": "image"},
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


class VisionClassifierCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        inputs = self.processor(
            images=[example["image"] for example in examples],
            return_tensors="pt",
            padding=True,
        )
        inputs["labels"] = torch.tensor(
            [example["label"] for example in examples],
            dtype=torch.long,
        )
        return inputs


class VisionAdapterClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()

        model_kwargs = {}
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16

        base_model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        base_model.config.use_cache = False
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=r".*vision_model.*self_attn.*(q_proj|k_proj|v_proj|out_proj)$",
        )
        self.model = get_peft_model(base_model, lora_config)

        for name, param in self.model.named_parameters():
            param.requires_grad = "vision_model" in name and "lora_" in name

        hidden_size = self.model.base_model.model.config.vision_config.hidden_size
        self.classifier = nn.Linear(hidden_size, len(LABEL_MAP))
        self.model.print_trainable_parameters()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def forward(self, pixel_values, labels=None, **kwargs):
        if pixel_values.dim() == 5:
            batch_size, num_images, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_images, channels, height, width)
        else:
            batch_size = pixel_values.shape[0]
            num_images = 1

        vision_dtype = next(self.model.base_model.model.model.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)
        vision_outputs = self.model.base_model.model.model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.last_hidden_state.mean(dim=1)
        pooled = pooled.reshape(batch_size, num_images, -1).mean(dim=1)
        pooled = pooled.to(dtype=self.classifier.weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels.long())

        return {"loss": loss, "logits": logits}

    def save_pretrained(self, output_dir):
        self.model.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), Path(output_dir) / "classifier_head.pt")


def print_one_eval_example(model, processor, split="test", index=0):
    dataset = load_dataset("thinkonward/reflection-connection", split=split)
    sample = dataset[index]
    label = LABEL_MAP[sample["label"]]
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
        generated_ids = model.generate(**inputs, max_new_tokens=80)
    prompt_len = inputs["input_ids"].shape[-1]
    generated_text = processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
    )[0].strip()

    print("one-example evaluation")
    print(f"source: {image_path}")
    print(f"label: {label}")
    print(f"generated: {generated_text}")


class DecoderLatentCollator:
    def __init__(self, processor):
        self.processor = processor
        self.domain_token_ids = self._build_domain_token_ids()

    def _build_domain_token_ids(self):
        token_ids = set()
        for word in DOMAIN_WORDS:
            encoded = self.processor.tokenizer(
                word,
                add_special_tokens=False,
            ).input_ids
            token_ids.update(encoded)
            encoded_with_space = self.processor.tokenizer(
                f" {word}",
                add_special_tokens=False,
            ).input_ids
            token_ids.update(encoded_with_space)
        return sorted(token_ids)

    def __call__(self, examples):
        # template
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
        # apply chat template
        prompts = [
            self.processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
            )
            for message in messages
        ]
        # give out pixel_values and input_ids of chat_template , each one have same template
        inputs = self.processor(
            text=prompts,
            images=[[example["image"]] for example in examples],
            return_tensors="pt",
            padding=True,
        )
        # attach label ids into dataset
        inputs["metric_labels"] = torch.tensor(
            [example["label"] for example in examples],
            dtype=torch.long,
        )
        inputs["domain_token_ids"] = torch.tensor(
            self.domain_token_ids,
            dtype=torch.long,
        )
        return inputs


def mean_pool_hidden(hidden_states, attention_mask=None):
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def coverage_loss(embeddings):
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())

    embeddings = F.normalize(embeddings, dim=-1)
    similarity = embeddings @ embeddings.T
    off_diagonal = similarity[~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)]
    return off_diagonal.pow(2).mean()


def token_level_alignment_loss(token_latents, vision_latent, attention_mask=None):
    token_latents = F.normalize(token_latents.float(), dim=-1)
    vision_latent = F.normalize(vision_latent.float(), dim=-1)
    distances = 1.0 - F.cosine_similarity(token_latents, vision_latent.unsqueeze(1), dim=-1)

    if attention_mask is None:
        return distances.mean()

    mask = attention_mask.to(distances.device, dtype=distances.dtype)
    return (distances * mask).sum() / mask.sum().clamp_min(1.0)


def output_steering_loss(
    logits,
    input_ids,
    domain_token_ids,
    eos_token_id=None,
    min_response_tokens=8,
    max_response_tokens=40,
):
    if logits.size(1) < 2:
        return logits.new_zeros(())

    next_token_logits = logits[:, :-1, :].float()
    next_token_ids = input_ids[:, 1:]
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    probs = log_probs.exp()

    sequence_length = next_token_logits.size(1)
    early_window = min(min_response_tokens, sequence_length)
    late_start = min(max_response_tokens, sequence_length)

    if eos_token_id is not None:
        eos_probs = probs[:, :, eos_token_id]
        early_eos_penalty = eos_probs[:, :early_window].mean()
        late_eos_reward = -0.05 * eos_probs[:, late_start:].mean() if late_start < sequence_length else logits.new_zeros(())
    else:
        early_eos_penalty = logits.new_zeros(())
        late_eos_reward = logits.new_zeros(())

    if domain_token_ids is not None and domain_token_ids.numel() > 0:
        domain_token_ids = domain_token_ids.to(next_token_logits.device)
        domain_prob = probs[:, :early_window, :].index_select(-1, domain_token_ids).sum(dim=-1)
        domain_reward = -torch.log(domain_prob.clamp_min(1e-8)).mean()
    else:
        domain_reward = logits.new_zeros(())

    repeated = next_token_ids[:, 1:] == next_token_ids[:, :-1]
    repetition_penalty = repeated.float().mean() if repeated.numel() > 0 else logits.new_zeros(())

    return early_eos_penalty + 0.2 * domain_reward + 0.1 * repetition_penalty + late_eos_reward


class DecoderLatentAlignmentModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, vision_adapter_path=OUTPUT_DIR / "final", latent_dim=512):
        super().__init__()

        base_model = AutoModelForImageTextToText.from_pretrained(model_name)
        vision_adapter_path = Path(vision_adapter_path)
        if vision_adapter_path.exists():
            vision_model = PeftModel.from_pretrained(
                base_model,
                vision_adapter_path.as_posix(),
                adapter_name="vision_adapter",
                is_trainable=False,
            )
            print(f"loaded frozen vision adapter from {vision_adapter_path}")
            self.model = vision_model.merge_and_unload()
            print("merged vision adapter into base model for decoder-adapter training")
        else:
            print(f"vision adapter not found at {vision_adapter_path}; using frozen base vision tower")
            self.model = base_model

        decoder_lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=["lm_head", "embed_tokens"],
        )
        self.model = get_peft_model(
            self.model,
            decoder_lora_config,
            adapter_name="decoder_adapter",
        )
        self.model.set_adapter("decoder_adapter")

        for name, param in self.model.named_parameters():
            param.requires_grad = "decoder_adapter" in name
            if "vision_model" in name or "connector" in name:
                param.requires_grad = False

        config = self.model.base_model.model.config
        # latents
        self.decoder_projector = nn.Linear(config.text_config.hidden_size, latent_dim)
        self.vision_projector = nn.Linear(config.vision_config.hidden_size, latent_dim)

        # skip vision encoder
        for param in self.vision_projector.parameters():
            param.requires_grad = False
        for param in self.decoder_projector.parameters():
            param.requires_grad = True
        # pairing decoder latent to encoder talent
        self.infonce_loss = NTXentLoss(temperature=0.07)
        # paring label to its latent anchor
        self.proxy_anchor_loss = ProxyAnchorLoss(
            num_classes=len(LABEL_MAP),
            embedding_size=latent_dim,
        )
        # softmax
        self.metric_softmax_loss = NormalizedSoftmaxLoss(
            num_classes=len(LABEL_MAP),
            embedding_size=latent_dim,
        )

        self.model.print_trainable_parameters()

    def get_eos_token_id(self):
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None and getattr(generation_config, "eos_token_id", None) is not None:
            return generation_config.eos_token_id

        config = getattr(self.model, "config", None)
        if config is not None:
            eos_token_id = getattr(config, "eos_token_id", None)
            if eos_token_id is not None:
                return eos_token_id

            text_config = getattr(config, "text_config", None)
            if text_config is not None and getattr(text_config, "eos_token_id", None) is not None:
                return text_config.eos_token_id

        return None

    def encode_vision_target(self, pixel_values):
        # batching or not
        if pixel_values.dim() == 5:
            batch_size, num_images, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_images, channels, height, width)
        else:
            batch_size = pixel_values.shape[0]
            num_images = 1
        # project to latents
        vision_dtype = next(self.model.base_model.model.model.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)
        vision_outputs = self.model.base_model.model.model.vision_model(pixel_values=pixel_values)
        vision_hidden = vision_outputs.last_hidden_state.mean(dim=1)
        vision_hidden = vision_hidden.reshape(batch_size, num_images, -1).mean(dim=1)
        vision_hidden = vision_hidden.to(dtype=self.vision_projector.weight.dtype)
        return self.vision_projector(vision_hidden)

    def forward(self, metric_labels=None, domain_token_ids=None, **inputs):
        vision_dtype = next(self.model.base_model.model.model.vision_model.parameters()).dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=vision_dtype)

        pixel_values = inputs["pixel_values"]
        attention_mask = inputs.get("attention_mask")

        with torch.no_grad():
            vision_latent = self.encode_vision_target(pixel_values)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        decoder_hidden = mean_pool_hidden(outputs.hidden_states[-1], attention_mask)
        decoder_hidden = decoder_hidden.to(dtype=self.decoder_projector.weight.dtype)
        decoder_latent = self.decoder_projector(decoder_hidden)
        token_hidden = outputs.hidden_states[-1].to(dtype=self.decoder_projector.weight.dtype)
        token_latents = self.decoder_projector(token_hidden)

        #normalise latents
        decoder_latent = F.normalize(decoder_latent.float(), dim=-1)
        vision_latent = F.normalize(vision_latent.float(), dim=-1).detach()

        # combine latents
        pair_embeddings = torch.cat([decoder_latent, vision_latent], dim=0)
        # matrices of size latents 2 dim
        pair_labels = torch.arange(decoder_latent.size(0), device=decoder_latent.device).repeat(2)

        # all losses
        infonce = self.infonce_loss(pair_embeddings, pair_labels)
        proxy_anchor = self.proxy_anchor_loss(decoder_latent, metric_labels)
        metric_cross_entropy = self.metric_softmax_loss(decoder_latent, metric_labels)
        coverage = coverage_loss(decoder_latent)

        cosine = 1.0 - F.cosine_similarity(decoder_latent, vision_latent, dim=-1).mean()
        token_alignment = token_level_alignment_loss(token_latents, vision_latent, attention_mask)
        output_steering = output_steering_loss(
            outputs.logits,
            inputs["input_ids"],
            domain_token_ids,
            eos_token_id=self.get_eos_token_id(),
        )

        loss = (
            infonce
            + proxy_anchor
            + metric_cross_entropy
            + 0.1 * coverage
            + cosine
            + 0.25 * token_alignment
            + 0.5 * output_steering
        )

        return {
            "loss": loss,
            "infonce_loss": infonce.detach(),
            "proxy_anchor_loss": proxy_anchor.detach(),
            "metric_cross_entropy_loss": metric_cross_entropy.detach(),
            "coverage_loss": coverage.detach(),
            "cosine_loss": cosine.detach(),
            "token_alignment_loss": token_alignment.detach(),
            "output_steering_loss": output_steering.detach(),
        }


class LatentAlignmentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

def train_decoder_with_label():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = VisionAdapterClassifier(MODEL_NAME)

    train_dataset = SeismicImageDataset("train")
    test_dataset = SeismicImageDataset("test")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR.as_posix(),
        logging_dir="logs",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
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
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=args,
        data_collator=VisionClassifierCollator(processor),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001,
            )
        ],
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model((OUTPUT_DIR / "final").as_posix())

def train_decoder_without_label():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = DecoderLatentAlignmentModel(
        model_name=MODEL_NAME,
        vision_adapter_path=OUTPUT_DIR / "final",
    )

    train_dataset = SeismicImageDataset("train")
    eval_dataset = SeismicImageDataset("test")

    args = TrainingArguments(
        output_dir=CUSTOM_OUTPUT_DIR.as_posix(),
        logging_dir="logs",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        weight_decay=0.02,
        warmup_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        prediction_loss_only=True,
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

    trainer.train()
    trainer.evaluate()
    trainer.save_model((CUSTOM_OUTPUT_DIR / "final").as_posix())
    print_one_eval_example(model.model, processor)

def main():
    train_decoder_with_label() # train vision-adapter and decoder with output label
    train_decoder_without_label()
# train using unsloth
# custom loss only for decoder layer
def test_understanding():
    test_dataset = load_dataset("thinkonward/reflection-connection", split="test")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    # test knowledge
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION}
            ]
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(text=prompt, images=[test_dataset[0]["image"].convert("RGB")], return_tensors="pt")

    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME)

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(generated_texts[0])

if __name__ == "__main__":
    main()
# 1. test model checked (not so good)
# 1.1 train decoder and vision-adapter (not mess with encoder because need shape and texture understanding) on label, maybe it can generate description
# 2. train model with custom loss to ensure in-topic generation with label (train just dec)
# 3. using finetune-vision that able to give out label (classification head output token?) wiring that to decoder instead for decoder only train without label
# introducing some new mechanics
