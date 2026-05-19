import argparse
import base64
import json
import mimetypes
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


DEFAULT_VLLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OUTPUT_DIR = Path("outputs/generated_unicamp_instructions")
K2_MODEL_DIR = Path("models/k2")
K2_TRAINED_PROJECTOR = Path("outputs/k2_attached_vision/final/k2_qwen_vision_projector.pt")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}

QUESTION_PROMPT = (
    "You are creating visual instruction data for 2D seismic interpretation. "
    "Look at this seismic image and write one answerable question about visible seismic "
    "patterns. The question must be specific to the image and should ask about reflector "
    "continuity, amplitude, geometry, faults, channels, chaos, or transparency. "
    "Return only the question."
)

ANSWER_PROMPT_TEMPLATE = (
    "You are a seismic interpretation assistant using thinking mode. "
    "Answer the question using only visible evidence from the seismic image. "
    "If auxiliary FaultNet detections are provided, use them as weak evidence, not as guaranteed truth. "
    "If no auxiliary detections are provided, do not assume faults are absent; judge the image directly.\n\n"
    "Question: {question}\n"
    "Auxiliary FaultNet detections: {detections}\n\n"
    "Return strict JSON only with keys: reasoning, final_answer. "
    "reasoning must be a list of 2-5 short evidence-grounded steps. "
    "final_answer must be a concise answer to the question. "
    "Do not include unsupported geology, well logs, depth, survey metadata, or exact locations unless visible or supported by detections."
)

K2_ANSWER_PROMPT_TEMPLATE = (
    "The preceding soft visual prefix comes from a Qwen-VL image encoder that processed "
    "a 2D seismic image. Answer this seismic interpretation question using visible image "
    "evidence only. Use concise geoscience language. Do not invent labels or structures.\n\n"
    "Question: {question}\n"
)

MERGE_PROMPT_TEMPLATE = (
    "You are creating the final supervised target for seismic vision-language training. "
    "Use the Qwen visual answer as the primary source for what is visible in the image. "
    "Use the K2 geoscience answer only to improve domain terminology and fluency when it is provided. "
    "FaultNet detections are weak extra evidence, not guaranteed truth. "
    "If FaultNet detections are none, this only means no auxiliary detector evidence is available; it does not rule out visible faults. "
    "Write an answer that is grounded in the image and avoids unsupported geology, well logs, survey metadata, "
    "exact locations, or claims not visible in the seismic section.\n\n"
    "Question: {question}\n"
    "Auxiliary FaultNet detections: {detections}\n"
    "Qwen visual reasoning: {qwen_reasoning}\n"
    "Qwen visual answer: {qwen_answer}\n"
    "K2 geoscience reasoning: {k2_reasoning}\n"
    "K2 geoscience answer: {k2_answer}\n\n"
    "Return strict JSON only with keys: reasoning, final_answer. "
    "reasoning must be a list of 2-4 short evidence-grounded steps. "
    "final_answer must be one concise seismic interpretation answer using visible evidence and domain language."
)


def load_image(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        array = np.squeeze(array)
        if array.ndim > 2:
            array = array[..., 0]
        low, high = np.percentile(array, [1, 99])
        array = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
        array = (array * 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    array = np.asarray(Image.open(path))
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[-1] in (3, 4) and array.dtype == np.uint8:
        return Image.fromarray(array[..., :3]).convert("RGB")
    if array.ndim == 3:
        array = array[..., 0]
    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError(f"Image has no finite pixel values: {path}")
    array = np.where(finite, array, np.nanmedian(array[finite]))
    low, high = np.percentile(array[finite], [1, 99])
    if abs(float(high - low)) < 1e-6:
        low, high = float(array[finite].min()), float(array[finite].max())
    array = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def iter_split_images(data_root, split):
    split_root = Path(data_root) / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split folder not found: {split_root}")
    for path in sorted(split_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def image_to_data_url(image, image_format="PNG"):
    with tempfile.NamedTemporaryFile(suffix=f".{image_format.lower()}") as file:
        image.save(file.name, format=image_format)
        data = Path(file.name).read_bytes()
    mime_type = mimetypes.types_map.get(f".{image_format.lower()}", "image/png")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class VLLMClient:
    def __init__(self, base_url=DEFAULT_VLLM_BASE_URL, model=DEFAULT_VLLM_MODEL, api_key="EMPTY"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate(self, image, prompt, max_new_tokens=256):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()


def parse_json_object(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def split_reasoning_answer(text):
    parsed = parse_json_object(text)
    if isinstance(parsed, dict):
        reasoning = parsed.get("reasoning", [])
        if isinstance(reasoning, str):
            reasoning = [reasoning]
        if not isinstance(reasoning, list):
            reasoning = []
        reasoning = [str(step).strip() for step in reasoning if str(step).strip()]
        final_answer = parsed.get("final_answer") or parsed.get("answer") or ""
        final_answer = str(final_answer).strip()
        if final_answer:
            return reasoning, final_answer
    return [], text.strip()


def format_reasoning_answer(reasoning, final_answer):
    if reasoning:
        reasoning_text = "\n".join(f"- {step}" for step in reasoning)
        return f"Reasoning:\n{reasoning_text}\nFinal answer: {final_answer}"
    return final_answer


def merge_answers(
    vllm_client,
    image,
    question,
    detections,
    qwen_reasoning,
    qwen_answer,
    k2_reasoning,
    k2_answer,
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
    raw = vllm_client.generate(image, prompt, max_new_tokens=384)
    reasoning, final_answer = split_reasoning_answer(raw)
    if not final_answer:
        final_answer = qwen_answer or k2_answer
    return raw, reasoning, final_answer


def load_faultnet(weights_path):
    if weights_path is None:
        return None
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "FaultNet YOLO inference needs `ultralytics`. Install it or omit --faultnet-weights."
        ) from error

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"FaultNet weights not found: {weights_path}")
    return YOLO(weights_path.as_posix())


def run_faultnet(model, image_path, conf=0.25):
    if model is None:
        return []
    image = np.asarray(load_image(image_path))
    results = model.predict(source=image, conf=conf, verbose=False)
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist() if getattr(boxes, "cls", None) is not None else [None] * len(xyxy)
        names = getattr(result, "names", {}) or {}
        masks = getattr(result, "masks", None)
        mask_data = masks.data.detach().cpu() if masks is not None and getattr(masks, "data", None) is not None else None
        polygons = masks.xy if masks is not None and getattr(masks, "xy", None) is not None else []
        for index, (box, score, class_id) in enumerate(zip(xyxy, confs, classes)):
            label = names.get(int(class_id), "fault") if class_id is not None else "fault"
            detection = {
                "label": str(label),
                "confidence": round(float(score), 4),
                "box_xyxy": [round(float(value), 2) for value in box],
            }
            if mask_data is not None and index < len(mask_data):
                mask = mask_data[index].float()
                detection["mask_coverage"] = round(float(mask.mean().item()), 6)
            if index < len(polygons) and len(polygons[index]) > 0:
                polygon = polygons[index]
                step = max(len(polygon) // 16, 1)
                detection["polygon_xy_sample"] = [
                    [round(float(x), 2), round(float(y), 2)]
                    for x, y in polygon[::step][:16]
                ]
            detections.append(detection)
    return detections


class K2VisionResponder:
    def __init__(self, k2_dir=K2_MODEL_DIR, projector_path=K2_TRAINED_PROJECTOR):
        import torch

        from k2_vision_pipeline import (
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
                "content": [{"type": "text", "text": format_reasoning_answer(merged_reasoning or [], merged_answer)}],
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
):
    output_path = Path(output_dir) / f"{split}.jsonl"
    sft_output_path = Path(output_dir) / f"{split}_sft.jsonl"
    count = 0
    for image_path in iter_split_images(data_root, split):
        if max_samples is not None and count >= max_samples:
            break

        image = load_image(image_path)
        detections = run_faultnet(faultnet, image_path, conf=faultnet_conf)
        detection_text = json.dumps(detections, ensure_ascii=False) if detections else "none"

        question = vllm_client.generate(
            image,
            QUESTION_PROMPT,
            max_new_tokens=96,
        )
        answer_prompt = ANSWER_PROMPT_TEMPLATE.format(
            question=question,
            detections=detection_text,
        )
        generator_answer_raw = vllm_client.generate(
            image,
            answer_prompt,
            max_new_tokens=384,
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
        )


if __name__ == "__main__":
    main()
