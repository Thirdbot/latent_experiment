import base64
import json
import mimetypes
import tempfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from seismic_k2.config import DEFAULT_VLLM_BASE_URL, DEFAULT_VLLM_MODEL
from seismic_k2.dataset_generator.image_utils import resize_for_vllm


def text_content(message):
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        if texts:
            return "\n".join(texts)

    text = message.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def maybe_add_reasoning(content, reasoning, include_reasoning):
    if not include_reasoning or not reasoning:
        return content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        if not parsed.get("reasoning"):
            parsed["reasoning"] = [reasoning]
        return json.dumps(parsed, ensure_ascii=False)

    return json.dumps(
        {
            "reasoning": [reasoning],
            "final_answer": content,
        },
        ensure_ascii=False,
    )


def extract_message_text(result, include_reasoning=False, allow_reasoning_as_content=False):
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError(f"vLLM response did not include choices: {json.dumps(result)[:1000]}")

    message = choices[0].get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str):
        reasoning = reasoning.strip()
    else:
        reasoning = ""

    content = text_content(message)
    if content:
        return maybe_add_reasoning(content, reasoning, include_reasoning)

    if isinstance(reasoning, str) and reasoning.strip():
        if allow_reasoning_as_content:
            return reasoning.strip()
        finish_reason = choices[0].get("finish_reason")
        raise RuntimeError(
            "vLLM returned reasoning text but no final assistant content "
            f"(finish_reason={finish_reason}). Disable reasoning for this endpoint or increase max_tokens. "
            f"reasoning_preview={reasoning.strip()[:500]!r}"
        )

    finish_reason = choices[0].get("finish_reason")
    raise RuntimeError(
        "vLLM returned an empty assistant message "
        f"(finish_reason={finish_reason}): {json.dumps(result)[:1000]}"
    )


def image_to_data_url(image, image_format="PNG"):
    image = resize_for_vllm(image)
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

    def generate(
        self,
        image,
        prompt,
        max_new_tokens=256,
        temperature=0.0,
        include_reasoning=False,
        allow_reasoning_as_content=False,
    ):
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
            "temperature": temperature,
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
        try:
            with urlopen(request) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {error.code}: {body}") from error
        return extract_message_text(
            result,
            include_reasoning=include_reasoning,
            allow_reasoning_as_content=allow_reasoning_as_content,
        )
