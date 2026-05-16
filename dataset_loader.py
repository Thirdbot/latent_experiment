from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# seismic-label dataset

ds = load_dataset("thinkonward/reflection-connection")

processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

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


def transform(example):
    label = LABEL_MAP[example["label"]]
    image = example["image"].convert("RGB")
    # get pixel value and input_ids ;return same format
    processed = processor(
        text=label,
        images=image,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )

    example["image"] = image
    example["label"] = label
    example["pixel_values"] = processed.pixel_values.squeeze(0)
    example["input_ids"] = processed.input_ids.squeeze(0)

    return example


class LoadData:
    def __init__(self, mode="train", image_key="pixel_values", text_key="input_ids", batch_size=4):
        self.image_key = image_key
        self.text_key = text_key
        self.mode = mode

        # transform
        self.transformed_dataset = ds[mode].map(transform)


