import argparse
import json
from pathlib import Path

from seismic_k2.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_MODEL,
    K2_MODEL_DIR,
    K2_TRAINED_PROJECTOR,
)
from seismic_k2.dataset_generator.image_utils import iter_split_images, load_image
from seismic_k2.dataset_generator.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    K2_ANSWER_PROMPT_TEMPLATE,
    MERGE_PROMPT_TEMPLATE,
    QUESTION_PROMPT,
)
from seismic_k2.dataset_generator.text_utils import format_reasoning_answer, split_reasoning_answer
from seismic_k2.dataset_generator.vllm_client import VLLMClient
from seismic_k2.fault.detection import load_faultnet, run_faultnet


class K2VisionResponder:
    def __init__(self, k2_dir=K2_MODEL_DIR, projector_path=K2_TRAINED_PROJECTOR):
        import torch

        from seismic_k2.vlm.k2_vision import (
            encode_image_with_qwen,
            generate_with_visual_prefix,
            get_hidden_size,
            load_k2,
            load_or_create_projector,
            load_qwen_vision_encoder,
        )

        self.torch = torch
        self.encode_image_with_qwen = encode_image_with_qwen
        self.generate_with_visual_prefix = generate_with_visual_prefix
        self.qwen_model, self.qwen_processor = load_qwen_vision_encoder()
        self.k2_model, self.k2_tokenizer = load_k2(k2_dir)
        vision_hidden_size = get_hidden_size(self.qwen_model)
        k2_hidden_size = get_hidden_size(self.k2_model)
        self.projector = load_or_create_projector(
            vision_hidden_size,
            k2_hidden_size,
            projector_path,
        )
        self.projector = self.projector.to(self.k2_model.get_input_embeddings().weight.device)
        self.projector.eval()

    def answer(self, image, question):
        vision_latent = self.encode_image_with_qwen(
            self.qwen_model,
            self.qwen_processor,
            image,
            token_drop_rate=0.0,
        )
        vision_latent = vision_latent.to(
            device=next(self.projector.parameters()).device,
            dtype=next(self.projector.parameters()).dtype,
        )
        with self.torch.no_grad():
            visual_prefix = self.projector(vision_latent)
        prompt = K2_ANSWER_PROMPT_TEMPLATE.format(question=question)
        return self.generate_with_visual_prefix(
            self.k2_model,
            self.k2_tokenizer,
            visual_prefix,
            prompt,
        )


def merge_answers(
    vllm_client,
    image,
    question,
    detections,
    qwen_reasoning,
    qwen_answer,
    k2_reasoning,
    k2_answer,
    temperature=0.0,
):
    detection_text = json.dumps(detections, ensure_ascii=False) if detections else "none"
    prompt = MERGE_PROMPT_TEMPLATE.format(
        question=question,
        detections=detection_text,
        qwen_reasoning=json.dumps(qwen_reasoning or [], ensure_ascii=False),
        qwen_answer=qwen_answer,
        k2_reasoning=json.dumps(k2_reasoning or [], ensure_ascii=False),
        k2_answer=k2_answer,
    )
    raw = vllm_client.generate(
        image,
        prompt,
        max_new_tokens=768,
        temperature=temperature,
        include_reasoning=True,
    )
    reasoning, final_answer = split_reasoning_answer(raw)
    if not final_answer:
        final_answer = qwen_answer or k2_answer
    return raw, reasoning, final_answer


def write_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sft_record(
    image_path,
    question,
    merged_reasoning,
    merged_answer,
    answer_source,
):
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": question},
        ],
    }
    return {
        "image": str(image_path),
        "question": question,
        "answer_source": answer_source,
        "reasoning": merged_reasoning or [],
        "answer": merged_answer,
        "messages": [
            user_message,
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": format_reasoning_answer(merged_reasoning or [], merged_answer),
                    }
                ],
            },
        ],
    }


def build_instruction_record(
    image_path,
    split,
    question,
    generator_answer,
    k2_answer,
    detections,
    generator_reasoning=None,
    k2_reasoning=None,
    generator_answer_raw=None,
    k2_answer_raw=None,
    merged_reasoning=None,
    merged_answer=None,
    merged_answer_raw=None,
):
    return {
        "image": str(image_path),
        "split": split,
        "question": question,
        "faultnet_detections": detections,
        "generator_answer_raw": generator_answer_raw or generator_answer,
        "generator_reasoning": generator_reasoning or [],
        "generator_answer": generator_answer,
        "k2_answer_raw": k2_answer_raw or k2_answer,
        "k2_reasoning": k2_reasoning or [],
        "k2_answer": k2_answer,
        "merged_answer_raw": merged_answer_raw or merged_answer,
        "merged_reasoning": merged_reasoning or [],
        "merged_answer": merged_answer,
        "messages_generator": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": generator_answer}],
            },
        ],
        "messages_k2": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": k2_answer}],
            },
        ],
        "messages_merged": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": format_reasoning_answer(merged_reasoning or [], merged_answer or ""),
                    }
                ],
            },
        ],
    }


def generate_split(
    split,
    data_root,
    output_dir,
    vllm_client,
    k2_responder=None,
    faultnet=None,
    faultnet_conf=0.05,
    max_samples=None,
    question_temperature=0.7,
    answer_temperature=0.0,
    merge_temperature=0.0,
):
    output_path = Path(output_dir) / f"{split}.jsonl"
    sft_output_path = Path(output_dir) / f"{split}_sft.jsonl"
    error_output_path = Path(output_dir) / f"{split}_errors.jsonl"
    count = 0
    for image_path in iter_split_images(data_root, split):
        if max_samples is not None and count >= max_samples:
            break

        try:
            image = load_image(image_path)
            detections = run_faultnet(faultnet, image_path, conf=faultnet_conf)
            detection_text = json.dumps(detections, ensure_ascii=False) if detections else "none"

            question = vllm_client.generate(
                image,
                QUESTION_PROMPT,
                max_new_tokens=256,
                temperature=question_temperature,
            )
            answer_prompt = ANSWER_PROMPT_TEMPLATE.format(
                question=question,
                detections=detection_text,
            )
            generator_answer_raw = vllm_client.generate(
                image,
                answer_prompt,
                max_new_tokens=768,
                temperature=answer_temperature,
                include_reasoning=True,
            )
            generator_reasoning, generator_answer = split_reasoning_answer(generator_answer_raw)
            if k2_responder is not None:
                k2_answer_raw = k2_responder.answer(image, question)
                k2_reasoning, k2_answer = split_reasoning_answer(k2_answer_raw)
                answer_source = "qwen_k2_merged"
            else:
                k2_answer_raw = ""
                k2_reasoning = []
                k2_answer = "K2 answer was not generated for this sample."
                answer_source = "qwen_faultnet_merged"

            merged_answer_raw, merged_reasoning, merged_answer = merge_answers(
                vllm_client=vllm_client,
                image=image,
                question=question,
                detections=detections,
                qwen_reasoning=generator_reasoning,
                qwen_answer=generator_answer,
                k2_reasoning=k2_reasoning,
                k2_answer=k2_answer,
                temperature=merge_temperature,
            )
            sft_record = build_sft_record(
                image_path=image_path,
                question=question,
                merged_reasoning=merged_reasoning,
                merged_answer=merged_answer,
                answer_source=answer_source,
            )
            record = build_instruction_record(
                image_path=image_path,
                split=split,
                question=question,
                generator_answer=generator_answer,
                k2_answer=k2_answer,
                detections=detections,
                generator_reasoning=generator_reasoning,
                k2_reasoning=k2_reasoning,
                generator_answer_raw=generator_answer_raw,
                k2_answer_raw=k2_answer_raw,
                merged_reasoning=merged_reasoning,
                merged_answer=merged_answer,
                merged_answer_raw=merged_answer_raw,
            )
            write_jsonl(output_path, record)
            write_jsonl(sft_output_path, sft_record)
            count += 1
            print(f"{split}[{count}] {image_path} merged={answer_source}")
        except Exception as error:
            error_record = {
                "image": str(image_path),
                "split": split,
                "error": str(error),
            }
            write_jsonl(error_output_path, error_record)
            print(f"{split}[skip] {image_path} error={error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/unicamp_namss",
        help="Root folder containing train/validation/test extracted Unicamp-NAMSS splits.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR.as_posix(),
        help="Output folder for generated JSONL instruction data.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--vllm-base-url",
        default=DEFAULT_VLLM_BASE_URL,
        help="OpenAI-compatible vLLM endpoint base URL.",
    )
    parser.add_argument(
        "--vllm-model",
        default=DEFAULT_VLLM_MODEL,
        help="Model name served by the vLLM endpoint.",
    )
    parser.add_argument(
        "--vllm-api-key",
        default="EMPTY",
        help="API key for the vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--faultnet-weights",
        default=None,
        help="Optional YOLO/FaultNet weights path. Omit to skip detection context.",
    )
    parser.add_argument(
        "--faultnet-conf",
        type=float,
        default=0.05,
        help="YOLO/FaultNet confidence threshold. Lower values are useful for Unicamp domain shift.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples per split for dry runs.",
    )
    parser.add_argument(
        "--question-temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for generated questions. Higher gives more diverse questions.",
    )
    parser.add_argument(
        "--answer-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for generated answers.",
    )
    parser.add_argument(
        "--merge-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for merged final SFT answers.",
    )
    parser.add_argument(
        "--k2-dir",
        default=K2_MODEL_DIR.as_posix(),
        help="Local K2 model repo folder.",
    )
    parser.add_argument(
        "--projector",
        default=K2_TRAINED_PROJECTOR.as_posix(),
        help="Qwen-vision-to-K2-prefix projector path.",
    )
    parser.add_argument(
        "--use-k2",
        action="store_true",
        help="Also load local Qwen vision + K2 responder. This needs extra GPU memory and may conflict with vLLM on one GPU.",
    )
    args = parser.parse_args()

    vllm_client = VLLMClient(
        base_url=args.vllm_base_url,
        model=args.vllm_model,
        api_key=args.vllm_api_key,
    )
    faultnet = load_faultnet(args.faultnet_weights)
    k2_responder = None
    if args.use_k2:
        k2_responder = K2VisionResponder(
            k2_dir=Path(args.k2_dir),
            projector_path=Path(args.projector),
        )

    for split in args.splits:
        generate_split(
            split=split,
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            vllm_client=vllm_client,
            k2_responder=k2_responder,
            faultnet=faultnet,
            faultnet_conf=args.faultnet_conf,
            max_samples=args.max_samples,
            question_temperature=args.question_temperature,
            answer_temperature=args.answer_temperature,
            merge_temperature=args.merge_temperature,
        )


if __name__ == "__main__":
    main()
