from pathlib import Path
import os
from datasets import load_dataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from unsloth import FastVisionModel, UnslothVisionDataCollator, is_bfloat16_supported
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

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

LABEL_ANSWER_MAP = {
    "Boring": "Boring. The image lacks a distinct amplitude anomaly, reflector offset, or clear channel-like geometry.",
    "Bright_Planar": "Bright_Planar. The image shows strong amplitudes arranged along relatively continuous planar reflectors.",
    "Bright_Chaotic": "Bright_Chaotic. The image shows high-amplitude reflections with chaotic texture and poor reflector continuity.",
    "Channel": "Channel. The image shows curved or incised reflector geometry consistent with a channel-like feature.",
    "Converging_Amplitudes": "Converging_Amplitudes. The image shows reflector amplitudes that narrow or converge across the section.",
    "Fault": "Fault. The image shows disrupted reflector continuity with offset-like geometry across a narrow zone.",
    "Salt": "Salt. The image shows a low-continuity body with chaotic or transparent seismic character around it.",
    "Transparent_Planar": "Transparent_Planar. The image shows weak amplitudes with planar layering and relatively transparent character.",
}

SEISMIC_CONCEPTS = [
    "low contrast",
    "high amplitude",
    "weak amplitude",
    "chaotic texture",
    "planar reflectors",
    "continuous reflectors",
    "poor reflector continuity",
    "disrupted reflectors",
    "offset geometry",
    "curved geometry",
    "incised channel",
    "converging amplitudes",
    "transparent character",
    "salt body",
    "layered structure",
    "amplitude anomaly",
]

LABEL_CONCEPT_MAP = {
    "Boring": ["low contrast", "weak amplitude"],
    "Bright_Planar": ["high amplitude", "planar reflectors", "continuous reflectors"],
    "Bright_Chaotic": ["high amplitude", "chaotic texture", "poor reflector continuity"],
    "Channel": ["curved geometry", "incised channel"],
    "Converging_Amplitudes": ["converging amplitudes", "amplitude anomaly"],
    "Fault": ["disrupted reflectors", "offset geometry", "poor reflector continuity"],
    "Salt": ["salt body", "chaotic texture", "transparent character"],
    "Transparent_Planar": ["weak amplitude", "transparent character", "planar reflectors"],
}

CONCEPT_INDEX = {concept: index for index, concept in enumerate(SEISMIC_CONCEPTS)}


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
        label = LABEL_MAP[sample["label"]]
        image = sample["image"].convert("RGB")
        concept_labels = torch.zeros(len(SEISMIC_CONCEPTS), dtype=torch.float32)
        for concept in LABEL_CONCEPT_MAP[label]:
            concept_labels[CONCEPT_INDEX[concept]] = 1.0
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
                        {"type": "text", "text": LABEL_ANSWER_MAP[label]},
                    ],
                },
            ],
            "images": [image],
            "concept_labels": concept_labels,
        }


def load_unsloth_vision_model(model_name=MODEL_NAME):
    model, processor = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    return model, processor


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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=48,
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


def get_assistant_token_mask(input_ids, labels, attention_mask=None):
    if labels is not None:
        mask = labels.ne(-100)
    elif attention_mask is not None:
        mask = attention_mask.bool()
    else:
        mask = torch.ones_like(input_ids, dtype=torch.bool)
    return mask


def build_language_codebook(vlm, tokenizer, concepts):
    embedding_layer = vlm.get_input_embeddings()
    embedding_weight = embedding_layer.weight.detach()
    vectors = []
    for concept in concepts:
        token_ids = tokenizer(
            concept,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(embedding_weight.device)
        concept_vector = embedding_weight.index_select(0, token_ids[0]).mean(dim=0)
        vectors.append(concept_vector.float())
    return F.normalize(torch.stack(vectors, dim=0), dim=-1)


class ConceptQuantizationCollator:
    def __init__(self, model, processor):
        self.base_collator = UnslothVisionDataCollator(model, processor)

    def __call__(self, examples):
        base_examples = [
            {
                "messages": example["messages"],
                "images": example["images"],
            }
            for example in examples
        ]
        batch = self.base_collator(base_examples)
        batch["concept_labels"] = torch.stack([example["concept_labels"] for example in examples])
        return batch


class ConceptQuantizedVLM(nn.Module):
    def __init__(self, vlm, processor, latent_dim=512):
        super().__init__()
        self.vlm = vlm
        self.processor = processor
        self.image_token_id = get_image_token_id(vlm, processor)

        vlm_hidden_size = get_hidden_size(vlm)
        self.vlm_vision_projector = nn.Linear(vlm_hidden_size, latent_dim)
        self.decoder_projector = nn.Linear(vlm_hidden_size, latent_dim)
        self.concept_query_projector = nn.Linear(vlm_hidden_size, vlm_hidden_size)
        self.concept_projector = nn.Linear(vlm_hidden_size, latent_dim)
        self.decoder_to_vision_gate = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.concept_temperature = 0.07
        self.register_buffer(
            "language_codebook",
            build_language_codebook(vlm, processor.tokenizer, SEISMIC_CONCEPTS),
            persistent=True,
        )

    def print_trainable_parameters(self):
        if hasattr(self.vlm, "print_trainable_parameters"):
            self.vlm.print_trainable_parameters()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.vlm.gradient_checkpointing_enable()
            else:
                self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        if hasattr(self.vlm, "enable_input_require_grads"):
            self.vlm.enable_input_require_grads()

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.vlm.save_pretrained(output_dir)
        torch.save(
            {
                "vlm_vision_projector": self.vlm_vision_projector.state_dict(),
                "decoder_projector": self.decoder_projector.state_dict(),
                "concept_query_projector": self.concept_query_projector.state_dict(),
                "concept_projector": self.concept_projector.state_dict(),
                "decoder_to_vision_gate": self.decoder_to_vision_gate.state_dict(),
                "language_codebook": self.language_codebook.detach().cpu(),
                "seismic_concepts": SEISMIC_CONCEPTS,
            },
            output_dir / "concept_quantizer_heads.pt",
        )
        torch.save(self.state_dict(), output_dir / "pytorch_model.bin")

    def _pool_by_mask(self, hidden_states, mask):
        mask = mask.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _language_quantize(self, qwen_vision_hidden):
        qwen_vision_hidden = qwen_vision_hidden.to(dtype=self.concept_query_projector.weight.dtype)
        concept_query = self.concept_query_projector(qwen_vision_hidden)
        concept_query = F.normalize(concept_query.float(), dim=-1)
        codebook = F.normalize(self.language_codebook.to(concept_query.device), dim=-1)
        concept_logits = concept_query @ codebook.T / self.concept_temperature
        concept_weights = F.softmax(concept_logits, dim=-1)
        quantized_hidden = concept_weights @ codebook
        quantized_hidden = quantized_hidden.to(dtype=self.concept_projector.weight.dtype)
        concept_latent = F.normalize(self.concept_projector(quantized_hidden).float(), dim=-1)
        return concept_latent, concept_logits, concept_weights

    def top_concepts(self, qwen_vision_hidden, k=4):
        _, _, concept_weights = self._language_quantize(qwen_vision_hidden)
        scores, indices = concept_weights.topk(k=min(k, concept_weights.size(-1)), dim=-1)
        return [
            [(SEISMIC_CONCEPTS[index], float(score)) for score, index in zip(row_scores, row_indices)]
            for row_scores, row_indices in zip(scores.detach().cpu(), indices.detach().cpu())
        ]

    def forward(self, concept_labels=None, **inputs):
        outputs = self.vlm(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels")

        image_mask = input_ids.eq(self.image_token_id)
        assistant_mask = get_assistant_token_mask(input_ids, labels, attention_mask)
        if attention_mask is not None:
            assistant_mask = assistant_mask & attention_mask.bool()

        qwen_vision_hidden = self._pool_by_mask(hidden_states, image_mask)
        decoder_hidden = self._pool_by_mask(hidden_states, assistant_mask)

        qwen_vision_hidden = qwen_vision_hidden.to(dtype=self.vlm_vision_projector.weight.dtype)
        decoder_hidden = decoder_hidden.to(dtype=self.decoder_projector.weight.dtype)
        qwen_vision_latent = F.normalize(self.vlm_vision_projector(qwen_vision_hidden).float(), dim=-1)
        decoder_latent = F.normalize(self.decoder_projector(decoder_hidden).float(), dim=-1)
        concept_latent, concept_logits, concept_weights = self._language_quantize(qwen_vision_hidden)

        fused_decoder_latent = F.normalize(
            self.decoder_to_vision_gate(
                torch.cat([decoder_latent, qwen_vision_latent.detach(), concept_latent.detach()], dim=-1).to(
                    dtype=self.decoder_to_vision_gate[0].weight.dtype
                )
            ).float(),
            dim=-1,
        )

        ce_loss = outputs.loss
        decoder_vision_loss = 1.0 - F.cosine_similarity(fused_decoder_latent, qwen_vision_latent.detach(), dim=-1).mean()
        image_concept_loss = 1.0 - F.cosine_similarity(qwen_vision_latent, concept_latent, dim=-1).mean()
        decoder_concept_loss = 1.0 - F.cosine_similarity(fused_decoder_latent, concept_latent.detach(), dim=-1).mean()
        concept_entropy_loss = -(concept_weights * concept_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        if concept_labels is not None:
            concept_labels = concept_labels.to(concept_logits.device, dtype=concept_logits.dtype)
            concept_supervision_loss = F.binary_cross_entropy_with_logits(concept_logits, concept_labels)
        else:
            concept_supervision_loss = ce_loss.new_zeros(())

        loss = (
            ce_loss
            + 0.08 * decoder_vision_loss
            + 0.12 * image_concept_loss
            + 0.12 * decoder_concept_loss
            + 0.08 * concept_supervision_loss
            + 0.01 * concept_entropy_loss
        )

        return {
            "loss": loss,
            "logits": outputs.logits,
            "ce_loss": ce_loss.detach(),
            "decoder_vision_loss": decoder_vision_loss.detach(),
            "image_concept_loss": image_concept_loss.detach(),
            "decoder_concept_loss": decoder_concept_loss.detach(),
            "concept_supervision_loss": concept_supervision_loss.detach(),
            "concept_entropy_loss": concept_entropy_loss.detach(),
        }


class ConceptQuantizedTrainer(Trainer):
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
    vlm, processor = load_unsloth_vision_model(MODEL_NAME)
    vlm = FastVisionModel.get_peft_model(
        vlm,
        finetune_vision_layers=True,
        finetune_language_layers=True,
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
    model = ConceptQuantizedVLM(vlm, processor)
    model.print_trainable_parameters()

    train_dataset = SeismicConversationDataset("train")
    test_dataset = SeismicConversationDataset("test")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR.as_posix(),
        logging_dir="logs",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
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

    trainer = ConceptQuantizedTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=args,
        data_collator=ConceptQuantizationCollator(vlm, processor),
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
    print_one_eval_example(model.vlm, processor)

def main():
    train_decoder_with_label()
# train using unsloth
# custom loss only for decoder layer
def test_understanding():
    test_dataset = load_dataset("thinkonward/reflection-connection", split="test")
    model, processor = load_unsloth_vision_model(MODEL_NAME)
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
    selected_data = test_dataset[10]
    label = LABEL_MAP[selected_data["label"]]
    print("test: ",label)
    inputs = processor(text=prompt, images=[selected_data["image"].convert("RGB")], return_tensors="pt")
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
# 1. test model checked (not so good)
# 1.1 train decoder and vision-adapter (not mess with encoder because need shape and texture understanding) on label, maybe it can generate description
# 2. train model with custom loss to ensure in-topic generation with label (train just dec)
# 3. using finetune-vision that able to give out label (classification head output token?) wiring that to decoder instead for decoder only train without label
# introducing some new mechanics
