import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from seismic_k2.vlm.dataset import ExportedMultimodalDataset


MODEL_ID = os.getenv("INTERNVL_MODEL_ID", "OpenGVLab/InternVL3-2B")
OUTPUT_DIR = Path(os.getenv("INTERNVL_OUTPUT_DIR", "outputs/internvl_seismic"))
TRAIN_DATA = Path(os.getenv("TRAIN_DATA", "data/splits/train.csv"))
EVAL_DATA = Path(os.getenv("EVAL_DATA", "data/splits/validate.csv"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "512"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2048"))
TRAIN_MASK_DECODER = os.getenv("TRAIN_MASK_DECODER", "0") == "1"
MASK_OUTPUT_SIZE = int(os.getenv("MASK_OUTPUT_SIZE", "256"))
DO_EVAL = os.getenv("DO_EVAL", "1") == "1"
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "internvl-seismic")
WANDB_MODE = os.getenv("WANDB_MODE", "online")
SEG_TOKEN = "<SEG>"


class InternVLCollator:
    def __init__(self, processor, mask_decoder_enabled=False):
        self.processor = processor
        self.mask_decoder_enabled = mask_decoder_enabled
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def __call__(self, examples):
        prompts = []
        images = []
        for example in examples:
            answer = example["answer"]
            if self.mask_decoder_enabled and float(example["mask_valid"].item()) > 0.5:
                answer = f"{answer}\n{SEG_TOKEN}"
            prompts.append(self.format_prompt(example["question"], answer))
            images.append(example["image"])

        batch = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        batch["target_masks"] = torch.stack([example["mask"] for example in examples])
        batch["mask_valid"] = torch.stack([example["mask_valid"] for example in examples])
        return batch

    def format_prompt(self, question, answer):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(messages, tokenize=False)
        return f"<image>\nQuestion: {question}\nAnswer: {answer}"


class OptionalMaskDecoder(nn.Module):
    # Custom part: InternVL already produces a multimodal hidden state.
    # When masks exist, the hidden state at <SEG> is projected to a mask grid.
    def __init__(self, hidden_size, output_size=MASK_OUTPUT_SIZE):
        super().__init__()
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size * output_size),
        )

    def forward(self, seg_hidden):
        logits = self.net(seg_hidden)
        return logits.view(seg_hidden.size(0), 1, self.output_size, self.output_size)


def main():
    output_dir = OUTPUT_DIR
    final_dir = output_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    mask_decoder_enabled = TRAIN_MASK_DECODER
    if mask_decoder_enabled:
        tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN]})

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": {"": 0} if torch.cuda.is_available() else None,
        "quantization_config": quantization_config if torch.cuda.is_available() else None,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    except ValueError:
        model = AutoModel.from_pretrained(MODEL_ID, **model_kwargs)
    if mask_decoder_enabled:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ExportedMultimodalDataset(TRAIN_DATA, max_image_side=MAX_IMAGE_SIDE, mask_output_size=MASK_OUTPUT_SIZE)
    eval_dataset = ExportedMultimodalDataset(EVAL_DATA, max_image_side=MAX_IMAGE_SIDE, mask_output_size=MASK_OUTPUT_SIZE)
    mask_count = sum(1 for row in train_dataset.records if row.get("mask_image") and Path(row["mask_image"]).exists())
    if mask_decoder_enabled and mask_count == 0:
        print("mask decoder disabled: no mask_image files found in train data.")
        mask_decoder_enabled = False

    mask_decoder = None
    seg_token_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN) if mask_decoder_enabled else None
    if mask_decoder_enabled:
        hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config.llm_config, "hidden_size")
        mask_decoder = OptionalMaskDecoder(hidden_size).to(next(model.parameters()).device)
        mask_decoder.train()

    collator = InternVLCollator(processor, mask_decoder_enabled=mask_decoder_enabled)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collator) if DO_EVAL else None

    trainable = [p for p in model.parameters() if p.requires_grad]
    if mask_decoder is not None:
        trainable += list(mask_decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.02)
    wandb_run = start_wandb()

    history = []
    global_step = 0
    grad_accum = 4
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, next(model.parameters()).device)
            outputs = model(
                **{k: v for k, v in batch.items() if k not in {"target_masks", "mask_valid"}},
                output_hidden_states=mask_decoder is not None,
                return_dict=True,
            )
            loss = outputs.loss
            if mask_decoder is not None:
                loss = loss + mask_loss(outputs, batch, mask_decoder, seg_token_id)
            (loss / grad_accum).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                metrics = {"step": global_step, "epoch": epoch + 1, "loss": float(loss.detach().cpu())}
                history.append(metrics)
                if wandb_run:
                    wandb_run.log(metrics, step=global_step)
                if global_step % 10 == 0:
                    print(json.dumps(metrics))

        if eval_loader is not None:
            metrics = {"step": global_step, "epoch": epoch + 1, "eval_loss": evaluate(model, eval_loader)}
            history.append(metrics)
            if wandb_run:
                wandb_run.log(metrics, step=global_step)
            print(json.dumps(metrics))

    if wandb_run:
        wandb_run.finish()
    save_outputs(final_dir, model, processor, mask_decoder, history)


def mask_loss(outputs, batch, mask_decoder, seg_token_id):
    valid = batch["mask_valid"].flatten() > 0.5
    if not valid.any():
        return outputs.loss.new_tensor(0.0)
    seg_mask = batch["input_ids"].eq(seg_token_id)
    if not seg_mask.any():
        return outputs.loss.new_tensor(0.0)
    positions = seg_mask.float().argmax(dim=1).long()
    rows = torch.arange(batch["input_ids"].size(0), device=batch["input_ids"].device)
    seg_hidden = outputs.hidden_states[-1][rows, positions]
    logits = mask_decoder(seg_hidden)
    target = batch["target_masks"].to(logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits[valid], target[valid])
    dice = dice_loss(logits[valid], target[valid])
    return bce + dice


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1e-6) / (denom + 1e-6)).mean()


def evaluate(model, loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, next(model.parameters()).device)
            outputs = model(**{k: v for k, v in batch.items() if k not in {"target_masks", "mask_valid"}}, return_dict=True)
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def move_batch(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def start_wandb():
    if os.getenv("WANDB_MODE", WANDB_MODE) == "disabled":
        return None
    import wandb

    return wandb.init(
        project=WANDB_PROJECT,
        name=f"internvl-seismic-{time.strftime('%Y%m%d-%H%M%S')}",
        mode=WANDB_MODE,
        dir=(OUTPUT_DIR / "wandb").as_posix(),
    )


def save_outputs(final_dir, model, processor, mask_decoder, history):
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir / "internvl_lora_adapter")
    processor.save_pretrained(final_dir / "processor")
    if mask_decoder is not None:
        torch.save(mask_decoder.state_dict(), final_dir / "mask_decoder.pt")
    with (final_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    print(f"saved InternVL training outputs to {final_dir}")


if __name__ == "__main__":
    main()
