from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"

INSTRUCTION = "You are an expert geophysicist.Using only these  Classify this seismic reflection image using one of: \
        (Boring, Bright_Planar, Bright_Chaotic, Channel, Converging_Amplitudes, Fault, Salt, Transparent_Planar). \
        Then give 2-3 short visual reasons based only on visible patterns. Do not invent details. \
        Describe accurately what you see in this image."

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


def main():
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
# 1.1 train decoder only on label, maybe it can generate description
# 2. train model with custom loss to ensure in-topic generation with label (train just dec)
# 3. using finetune-vision that able to give out label (classification head output token?) wiring that to decoder instead for decoder only train without label
# introducing some new mechanics
