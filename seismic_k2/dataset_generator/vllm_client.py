import base64
import json
import mimetypes
import tempfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from seismic_k2.config import DEFAULT_VLLM_BASE_URL, DEFAULT_VLLM_MODEL
from seismic_k2.dataset_generator.image_utils import resize_for_vllm


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

    def generate(self, image, prompt, max_new_tokens=256, temperature=0.0):
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
        return result["choices"][0]["message"]["content"].strip()
