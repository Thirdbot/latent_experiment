from pathlib import Path

import numpy as np
from PIL import Image

from seismic_k2.config import IMAGE_SUFFIXES, VLLM_MAX_IMAGE_SIDE


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


def resize_for_vllm(image, max_side=VLLM_MAX_IMAGE_SIDE):
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image
    scale = max_side / largest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.BICUBIC)

