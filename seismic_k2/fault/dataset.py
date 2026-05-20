import argparse
import csv
import re
import struct
import warnings
import zipfile
import zlib
from ast import literal_eval
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


DATAVERSE_SERVER = "https://dataverse.harvard.edu"
DATAVERSE_PID = "doi:10.7910/DVN/YBYGBK"


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def get_dataset_files(server=DATAVERSE_SERVER, pid=DATAVERSE_PID):
    response = requests.get(
        f"{server}/api/datasets/:persistentId/",
        params={"persistentId": pid},
        timeout=60,
    )
    response.raise_for_status()
    files = response.json()["data"]["latestVersion"]["files"]
    rows = []
    for item in files:
        data_file = item["dataFile"]
        name = item.get("label") or data_file.get("filename")
        rows.append(
            {
                "id": data_file["id"],
                "name": name,
                "size": data_file.get("filesize", 0),
                "restricted": data_file.get("restricted", False),
            }
        )
    return rows


def classify_file(name):
    path = Path(name)
    if path.suffix.lower() not in {".npz", ".npy"}:
        return None, None

    stem = path.stem.lower()
    if stem.startswith("seis"):
        kind = "seis"
        key = re.sub(r"^seis(?:mic)?[_\-\s]*", "", stem)
    elif stem.startswith("fault"):
        kind = "fault"
        key = re.sub(r"^faults?[_\-\s]*", "", stem)
    else:
        return None, None
    return kind, re.sub(r"[^a-z0-9]+", "", key)


def choose_best_files(files):
    selected = {}
    for file in files:
        kind, key = classify_file(file["name"])
        if kind is None:
            continue
        pair_key = (kind, key)
        ext = Path(file["name"]).suffix.lower()
        if pair_key not in selected:
            selected[pair_key] = file
            continue
        old_ext = Path(selected[pair_key]["name"]).suffix.lower()
        if old_ext == ".npy" and ext == ".npz":
            selected[pair_key] = file
    return list(selected.values())


def write_split_csv(rows, output_dir, split):
    path = Path(output_dir) / f"{split}.csv"
    values = list(rows.values())
    values.sort(key=lambda row: int(row["id"]))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "seis", "fault"])
        writer.writeheader()
        for row in values:
            if row.get("seis") and row.get("fault"):
                writer.writerow(row)


def download_fault_dataset(output_dir, server=DATAVERSE_SERVER, pid=DATAVERSE_PID):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {"train": {}, "test": {}, "val": {}}

    files = choose_best_files(get_dataset_files(server=server, pid=pid))
    for file in progress(files, desc="Checking files"):
        kind, key = classify_file(file["name"])
        if kind is None:
            continue

        numbers = re.findall(r"\d+", key)
        if not numbers:
            continue
        number = numbers[0]
        file_name = Path(file["name"]).name
        file_path = output_dir / file_name

        if not file_path.exists():
            url = f"{server}/api/access/datafile/{file['id']}"
            print(f"download: {file_name}")
            with requests.get(url, stream=True, timeout=600) as response:
                response.raise_for_status()
                chunks = response.iter_content(chunk_size=1024 * 1024)
                total = file["size"] // (1024 * 1024) if file["size"] else None
                with file_path.open("wb") as out:
                    for chunk in progress(chunks, total=total, unit="MB", desc=file_name, leave=False):
                        if chunk:
                            out.write(chunk)
        else:
            print(f"skip: {file_name}")

        if "train" in key:
            split = "train"
        elif "test" in key:
            split = "test"
        elif "val" in key:
            split = "val"
        else:
            continue

        rows = split_rows[split]
        rows.setdefault(number, {"id": number, "seis": None, "fault": None})
        rows[number][kind] = str(file_path)

    for split, rows in split_rows.items():
        write_split_csv(rows, output_dir, split)


class DeflateReader:
    def __init__(self, file):
        self.file = file
        self.decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        self.buffer = b""

    def read(self, size=-1):
        if size < 0:
            chunks = [self.buffer]
            self.buffer = b""
            for chunk in iter(lambda: self.file.read(1024 * 1024), b""):
                chunks.append(self.decompressor.decompress(chunk))
            chunks.append(self.decompressor.flush())
            return b"".join(chunks)

        while len(self.buffer) < size:
            chunk = self.file.read(1024 * 1024)
            if not chunk:
                self.buffer += self.decompressor.flush()
                break
            self.buffer += self.decompressor.decompress(chunk)
        data = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return data


@contextmanager
def open_npy_member(path):
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as file:
                yield file
                return
    except zipfile.BadZipFile:
        pass

    with path.open("rb") as file:
        signature = file.read(4)
        if signature == b"\x93NUM":
            file.seek(0)
            yield file
            return
        if signature != b"PK\x03\x04":
            raise ValueError(f"{path} is not a .npy file or .npz archive")

        local_header = signature + file.read(26)
        _, _, _, compression, _, _, _, _, _, name_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH", local_header
        )
        file.seek(name_len + extra_len, 1)
        if compression == 0:
            yield file
        elif compression == 8:
            yield DeflateReader(file)
        else:
            raise ValueError(f"{path} uses unsupported zip compression method {compression}")


def read_npy_header(file, source_name):
    if file.read(6) != b"\x93NUMPY":
        raise ValueError(f"{source_name} does not contain a .npy array")
    version = file.read(2)
    header_size_bytes = 2 if version == b"\x01\x00" else 4
    header_size_format = "<H" if header_size_bytes == 2 else "<I"
    header_size = struct.unpack(header_size_format, file.read(header_size_bytes))[0]
    header = file.read(header_size).decode("latin1")
    shape_text = header.split("'shape':", 1)[1].split(")", 1)[0] + ")"
    dtype_text = header.split("'descr':", 1)[1].split(",", 1)[0].strip().strip("'\"")
    return literal_eval(shape_text), np.dtype(dtype_text)


def to_uint8(array):
    array = array.astype(np.float32, copy=False)
    low, high = np.percentile(array, (1, 99))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    array = np.clip((array - low) / (high - low), 0, 1)
    return (array * 255).astype(np.uint8)


def write_yolo_label(label_path, fault_mask):
    height, width = fault_mask.shape
    mask = (fault_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(1.0, 0.001 * perimeter)
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        if len(polygon) < 3:
            polygon = contour.reshape(-1, 2)
        if len(polygon) < 3:
            continue

        coords = []
        for x, y in polygon:
            coords.append(f"{min(max(float(x) / width, 0.0), 1.0):.6f}")
            coords.append(f"{min(max(float(y) / height, 0.0), 1.0):.6f}")
        lines.append("0 " + " ".join(coords))
    Path(label_path).write_text("\n".join(lines) + ("\n" if lines else ""))


class YOLOFormatConverter:
    def __init__(self, dataset_dir, output_dir):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.images_folder = self.output_dir / "images"
        self.labels_folder = self.output_dir / "labels"
        self.yaml_path = self.output_dir / "data.yaml"

    def prepare_folders(self):
        for split in ("train", "test", "val"):
            for folder in (self.images_folder / split, self.labels_folder / split):
                folder.mkdir(parents=True, exist_ok=True)
                for old_file in folder.glob("*"):
                    if old_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".txt"}:
                        old_file.unlink()

    def write_data_yaml(self):
        self.yaml_path.write_text(
            "\n".join(
                [
                    f"path: {self.output_dir.absolute()}",
                    "train: images/train",
                    "val: images/val",
                    "test: images/test",
                    "",
                    "names:",
                    "  0: fault",
                    "",
                ]
            )
        )

    def resolve_dataset_path(self, value):
        path = Path(value)
        if path.exists():
            return path
        return self.dataset_dir.parent / path

    def extract_pair(self, seismic_path, fault_path, split, max_slices=None):
        seismic_path = self.resolve_dataset_path(seismic_path)
        fault_path = self.resolve_dataset_path(fault_path)
        with open_npy_member(seismic_path) as seismic_file:
            with open_npy_member(fault_path) as fault_file:
                seismic_shape, seismic_dtype = read_npy_header(seismic_file, seismic_path.as_posix())
                fault_shape, fault_dtype = read_npy_header(fault_file, fault_path.as_posix())
                if seismic_shape != fault_shape:
                    raise ValueError(f"Shape mismatch: {seismic_path} {seismic_shape} vs {fault_path} {fault_shape}")
                if len(seismic_shape) != 3:
                    raise ValueError(f"{seismic_path} shape {seismic_shape} is not a 3D volume")

                file_name = seismic_path.stem
                section_shape = seismic_shape[1:]
                seismic_bytes = int(np.prod(section_shape)) * seismic_dtype.itemsize
                fault_bytes = int(np.prod(section_shape)) * fault_dtype.itemsize
                slice_count = seismic_shape[0] if max_slices is None else min(seismic_shape[0], max_slices)

                for idx in range(slice_count):
                    seismic_section = seismic_file.read(seismic_bytes)
                    fault_section = fault_file.read(fault_bytes)
                    if len(seismic_section) != seismic_bytes or len(fault_section) != fault_bytes:
                        warnings.warn(f"Skipping incomplete slice {idx} for {seismic_path}", stacklevel=2)
                        break
                    image_stem = f"{file_name}_{idx:03d}"
                    seismic = np.frombuffer(seismic_section, dtype=seismic_dtype).reshape(section_shape)
                    fault = np.frombuffer(fault_section, dtype=fault_dtype).reshape(section_shape)
                    Image.fromarray(to_uint8(seismic), mode="L").save(self.images_folder / split / f"{image_stem}.png")
                    write_yolo_label(self.labels_folder / split / f"{image_stem}.txt", fault)

    def read_split(self, split, max_slices=None):
        csv_path = self.dataset_dir / f"{split}.csv"
        with csv_path.open(newline="", encoding="utf-8") as file:
            for row in progress(csv.DictReader(file), desc=f"Convert {split}"):
                self.extract_pair(row["seis"], row["fault"], split=split, max_slices=max_slices)

    def create(self, max_slices=None):
        self.prepare_folders()
        self.write_data_yaml()
        for split in ("train", "test", "val"):
            self.read_split(split, max_slices=max_slices)
        return self.yaml_path


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--output-dir", default="data/fault_dataset")
    download_parser.add_argument("--server", default=DATAVERSE_SERVER)
    download_parser.add_argument("--pid", default=DATAVERSE_PID)

    convert_parser = subparsers.add_parser("convert-yolo")
    convert_parser.add_argument("--dataset-dir", default="data/fault_dataset")
    convert_parser.add_argument("--output-dir", default="data/fault_yolo")
    convert_parser.add_argument("--max-slices", type=int, default=None)

    args = parser.parse_args()
    if args.command == "download":
        download_fault_dataset(args.output_dir, server=args.server, pid=args.pid)
    elif args.command == "convert-yolo":
        converter = YOLOFormatConverter(args.dataset_dir, args.output_dir)
        yaml_path = converter.create(max_slices=args.max_slices)
        print(f"wrote YOLO dataset config: {yaml_path}")


if __name__ == "__main__":
    main()
