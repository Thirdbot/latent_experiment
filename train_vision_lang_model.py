from pathlib import Path
from datasets import load_dataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from pytorch_metric_learning.losses import NTXentLoss, NormalizedSoftmaxLoss, ProxyAnchorLoss

MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"

INSTRUCTION = "You are an expert geophysicist.Using only these  Classify this seismic reflection image using one of: \
        (Boring, Bright_Planar, Bright_Chaotic, Channel, Converging_Amplitudes, Fault, Salt, Transparent_Planar). \
        Then give 2-3 short visual reasons based only on visible patterns. Do not invent details. \
        Describe accurately what you see in this image."

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


class DecoderLatentCollator:
    def __init__(self, processor):
        self.processor = processor

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


class DecoderLatentAlignmentModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, latent_dim=512):
        super().__init__()

        self.model = AutoModelForImageTextToText.from_pretrained(model_name)

        lora_config = LoraConfig(
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
        self.model = get_peft_model(self.model, lora_config)

        for name, param in self.model.named_parameters():
            # skip vision model
            if "vision_model" in name or "connector" in name:
                param.requires_grad = False

        config = self.model.base_model.model.config
        # latents
        self.decoder_projector = nn.Linear(config.text_config.hidden_size, latent_dim)
        self.vision_projector = nn.Linear(config.vision_config.hidden_size, latent_dim)

        # skip vision encoder
        for param in self.vision_projector.parameters():
            param.requires_grad = False
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

    def encode_vision_target(self, pixel_values):
        # batching or not
        if pixel_values.dim() == 5:
            batch_size, num_images, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_images, channels, height, width)
        else:
            batch_size = pixel_values.shape[0]
            num_images = 1
        # project to latents
        vision_outputs = self.model.base_model.model.model.vision_model(pixel_values=pixel_values)
        vision_hidden = vision_outputs.last_hidden_state.mean(dim=1)
        vision_hidden = vision_hidden.reshape(batch_size, num_images, -1).mean(dim=1)
        return self.vision_projector(vision_hidden)

    def forward(self, metric_labels=None, **inputs):
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
        decoder_latent = self.decoder_projector(decoder_hidden)

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

        loss = (
            infonce
            + proxy_anchor
            + metric_cross_entropy
            + 0.1 * coverage
            + cosine
        )

        return {
            "loss": loss,
            "infonce_loss": infonce.detach(),
            "proxy_anchor_loss": proxy_anchor.detach(),
            "metric_cross_entropy_loss": metric_cross_entropy.detach(),
            "coverage_loss": coverage.detach(),
            "cosine_loss": cosine.detach(),
        }


class LatentAlignmentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

def train_decoder_with_label():
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Keep this as a Python dataset so Arrow does not try to serialize nested chat messages.
    train_conversations = SeismicConversationDataset("train")
    test_conversations = SeismicConversationDataset("test")

    args = SFTConfig(
        output_dir=OUTPUT_DIR.as_posix(),
        logging_dir="logs",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        weight_decay=0.02,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        max_length=None,
    )
    # self-supervised training with label to train decoder. will change to custom trainer for training decoder
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_conversations,
        eval_dataset=test_conversations,
        args=args,
        processing_class=processor,
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model((OUTPUT_DIR / "final").as_posix())

def train_decoder_without_label():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = DecoderLatentAlignmentModel(MODEL_NAME)

    train_dataset = SeismicImageDataset("train")
    eval_dataset = SeismicImageDataset("test")

    args = TrainingArguments(
        output_dir=CUSTOM_OUTPUT_DIR.as_posix(),
        logging_dir="logs",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
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
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model((CUSTOM_OUTPUT_DIR / "final").as_posix())


def main():
    train_decoder_with_label() # train vision-adapter and decoder with output label

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
