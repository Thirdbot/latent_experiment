from pathlib import Path
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from unsloth import FastVisionModel, UnslothVisionDataCollator, is_bfloat16_supported


TEACHER_MODEL_NAME = "daven3/k2"
STUDENT_MODEL_NAME = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
VISION_ADAPTER_DIR = Path("outputs/vision_llm_trained/final")
OUTPUT_DIR = Path("outputs/vision_llm_distilled")
NUM_STUDENT_CANDIDATES = 4

INSTRUCTION = (
    "You are an expert geophysicist interpreting seismic reflection images. "
    "Classify the image using exactly one of: Boring, Bright_Planar, Bright_Chaotic, "
    "Channel, Converging_Amplitudes, Fault, Salt, Transparent_Planar. "
    "Answer with exactly this format: <Class>. <one concise visible seismic reason>. "
    "Use English only and do not use markdown, bullets, numbering, or extra explanation."
)

TEACHER_PROMPT = (
    "You are a geophysicist writing concise seismic interpretation answers. "
    "Given the seismic class '{label}', write exactly one sentence in this format: "
    "{label}. <one concise visible seismic reason>. "
    "Use terms like reflector continuity, amplitude, geometry, offset, chaos, or transparency. "
    "Do not use markdown, bullets, uncertainty language, or extra explanation."
)

TEACHER_SCORING_PROMPT = (
    "You are a strict geophysics answer evaluator.\n"
    "True seismic class: {label}\n"
    "Student answer: {answer}\n\n"
    "Score the student answer from 0 to 5 using this rubric:\n"
    "1 point for starting with the correct class.\n"
    "1 point for being concise and one sentence.\n"
    "1 point for using plausible seismic terminology.\n"
    "1 point for giving a visible-reason style explanation.\n"
    "1 point for avoiding unsupported extra claims, markdown, or rambling.\n\n"
    "Return only this format: Score: <number>"
)

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


def load_student():
    model, processor = FastVisionModel.from_pretrained(
        STUDENT_MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    if VISION_ADAPTER_DIR.exists():
        vision_model = PeftModel.from_pretrained(
            model,
            VISION_ADAPTER_DIR.as_posix(),
            is_trainable=False,
        )
        model = vision_model.merge_and_unload()
        print(f"merged vision adapter from {VISION_ADAPTER_DIR}")
    else:
        print(f"vision adapter not found at {VISION_ADAPTER_DIR}; using base student vision tower")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
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

    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name or "merger" in name:
            param.requires_grad = False
        if "lm_head" in name or "embed_tokens" in name:
            param.requires_grad = False

    return model, processor


class TeacherAnswerGenerator:
    def __init__(self, model_name=TEACHER_MODEL_NAME):
        dtype = torch.bfloat16 if torch.cuda.is_available() and is_bfloat16_supported() else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.cache = {}
        self.score_cache = {}

    def __call__(self, label):
        if label in self.cache:
            return self.cache[label]

        prompt = TEACHER_PROMPT.format(label=label)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            )
        answer = self.tokenizer.decode(
            generated_ids[0, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        if not answer.startswith(label):
            answer = f"{label}. {answer}"
        answer = " ".join(answer.split())
        self.cache[label] = answer
        return answer

    def score(self, label, answer):
        cache_key = (label, answer)
        if cache_key in self.score_cache:
            return self.score_cache[cache_key]

        prompt = TEACHER_SCORING_PROMPT.format(label=label, answer=answer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
            )
        text = self.tokenizer.decode(
            generated_ids[0, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        match = re.search(r"([0-5](?:\.\d+)?)", text)
        score = float(match.group(1)) if match else 0.0
        self.score_cache[cache_key] = score
        return score


def generate_student_candidates(student, processor, image, num_candidates=NUM_STUDENT_CANDIDATES):
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
    device = next(student.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    was_training = student.training
    student.eval()
    with torch.no_grad():
        generated_ids = student.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=num_candidates,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    if was_training:
        student.train()

    prompt_len = inputs["input_ids"].shape[-1]
    candidates = processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
    )
    candidates = [" ".join(candidate.split()) for candidate in candidates if candidate.strip()]
    return list(dict.fromkeys(candidates))


def build_rejection_sampling_records(split, student, processor, teacher):
    dataset = load_dataset("thinkonward/reflection-connection", split=split)
    records = []
    for index, sample in enumerate(dataset):
        label = LABEL_MAP[sample["label"]]
        image = sample["image"].convert("RGB")
        candidates = generate_student_candidates(student, processor, image)
        candidates.append(teacher(label))

        scored_candidates = [
            (teacher.score(label, candidate), candidate)
            for candidate in candidates
        ]
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_answer = scored_candidates[0]
        print(f"{split}[{index}] label={label} teacher_score={best_score:.2f} answer={best_answer}")
        records.append(
            {
                "image": image,
                "answer": best_answer,
            }
        )
    return records


class DistillationDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
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
                        {"type": "text", "text": record["answer"]},
                    ],
                },
            ],
            "images": [record["image"]],
        }


def train_distillation():
    student, processor = load_student()
    teacher = TeacherAnswerGenerator()

    train_records = build_rejection_sampling_records("train", student, processor, teacher)
    eval_records = build_rejection_sampling_records("test", student, processor, teacher)
    train_dataset = DistillationDataset(train_records)
    eval_dataset = DistillationDataset(eval_records)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR.as_posix(),
        logging_dir="logs",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.02,
        warmup_steps=25,
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
        model=student,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=UnslothVisionDataCollator(student, processor),
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


if __name__ == "__main__":
    train_distillation()
