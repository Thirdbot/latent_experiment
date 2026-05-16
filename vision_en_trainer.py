from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, SiglipModel, TrainingArguments, Trainer

from dataset_loader import LABEL_MAP, LoadData


MODEL_NAME = "google/siglip-so400m-patch14-384"
OUTPUT_DIR = Path("outputs/vision_en_trained")
NUM_LABELS = len(LABEL_MAP)


class SiglipVisionClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.siglip = SiglipModel.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
            )
            self.siglip = prepare_model_for_kbit_training(self.siglip)
        else:
            self.siglip = SiglipModel.from_pretrained(MODEL_NAME)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=r".*vision_model.*self_attn.*(q_proj|k_proj|v_proj|out_proj)$",
        )
        self.siglip = get_peft_model(self.siglip, lora_config)
        hidden_size = self.siglip.base_model.model.config.vision_config.hidden_size
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)

        self.siglip.print_trainable_parameters()

    def forward(self, pixel_values, labels=None):
        vision_outputs = self.siglip.base_model.model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        logits = self.classifier(pooled.to(self.classifier.weight.dtype))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels.long())

        return {"loss": loss, "logits": logits}


def build_dataset(split):
    dataset = LoadData(split).transformed_dataset
    keep_columns = ["pixel_values", "label"] # because encoder need processed image and classification head needed it
    remove_columns = [name for name in dataset.column_names if name not in keep_columns]
    dataset = dataset.remove_columns(remove_columns)
    dataset.set_format(type="torch", columns=keep_columns)
    return dataset


args = TrainingArguments(
    output_dir=OUTPUT_DIR.as_posix(),
    logging_dir="logs",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.02,
    warmup_ratio=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
)


def main():
    trainer = Trainer(
        model=SiglipVisionClassifier(),
        args=args,
        train_dataset=build_dataset("train"),
        eval_dataset=build_dataset("test"),
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model((OUTPUT_DIR / "final").as_posix())


if __name__ == "__main__":
    main()
