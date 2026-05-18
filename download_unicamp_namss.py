import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RECORD_ID = "18330487"
ZENODO_API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
DEFAULT_OUTPUT_DIR = Path("data/unicamp_namss")
CHUNK_SIZE = 1024 * 1024 * 8


def fetch_record_metadata():
    request = Request(ZENODO_API_URL, headers={"User-Agent": "latent-experiment-downloader"})
    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Could not fetch Zenodo metadata from {ZENODO_API_URL}: {error}") from error


def get_files(metadata):
    files = []
    for file_info in metadata.get("files", []):
        key = file_info["key"]
        if key not in {"train.tar.gz", "validation.tar.gz", "test.tar.gz"}:
            continue
        checksum = file_info.get("checksum", "")
        md5 = checksum.replace("md5:", "") if checksum.startswith("md5:") else checksum
        links = file_info.get("links", {})
        download_url = links.get("self") or links.get("download")
        files.append(
            {
                "name": key,
                "size": int(file_info.get("size", 0)),
                "md5": md5,
                "url": download_url,
            }
        )
    return sorted(files, key=lambda item: item["name"])


def md5sum(path):
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_bytes(size):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def download_file(file_info, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / file_info["name"]
    expected_size = file_info["size"]
    existing_size = destination.stat().st_size if destination.exists() else 0

    if destination.exists() and expected_size and existing_size == expected_size:
        if file_info["md5"] and md5sum(destination) == file_info["md5"]:
            print(f"already downloaded and verified: {destination}")
            return destination
        print(f"existing file failed MD5 check, redownloading: {destination}")
        destination.unlink()
        existing_size = 0

    headers = {"User-Agent": "latent-experiment-downloader"}
    mode = "wb"
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
        print(f"resuming {file_info['name']} from {human_bytes(existing_size)}")
    else:
        print(f"downloading {file_info['name']} ({human_bytes(expected_size)})")

    request = Request(file_info["url"], headers=headers)
    try:
        with urlopen(request) as response, destination.open(mode) as file:
            downloaded = existing_size
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if expected_size:
                    pct = downloaded / expected_size * 100
                    print(
                        f"\r{file_info['name']}: {human_bytes(downloaded)} / "
                        f"{human_bytes(expected_size)} ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
            print()
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Could not download {file_info['name']}: {error}") from error

    if file_info["md5"]:
        actual_md5 = md5sum(destination)
        if actual_md5 != file_info["md5"]:
            raise RuntimeError(
                f"MD5 mismatch for {destination}: expected {file_info['md5']}, got {actual_md5}"
            )
        print(f"verified MD5: {destination}")

    return destination


def extract_archive(archive_path, output_dir):
    extract_dir = output_dir / archive_path.name.replace(".tar.gz", "")
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive_path} -> {extract_dir}")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extract_dir)
    return extract_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download Unicamp-NAMSS 2D seismic dataset from Zenodo record 18330487."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR.as_posix(),
        help="Directory to store downloaded archives.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test", "all"],
        default=["all"],
        help="Dataset splits to download.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract each downloaded .tar.gz archive.",
    )
    args = parser.parse_args()

    selected = {"train", "validation", "test"} if "all" in args.splits else set(args.splits)
    output_dir = Path(args.output_dir)

    metadata = fetch_record_metadata()
    files = [
        file_info
        for file_info in get_files(metadata)
        if file_info["name"].replace(".tar.gz", "") in selected
    ]
    if not files:
        raise RuntimeError("No matching files found in Zenodo metadata.")

    print("record:", metadata.get("title", RECORD_ID))
    print("output:", output_dir)
    for file_info in files:
        archive_path = download_file(file_info, output_dir)
        if args.extract:
            extract_archive(archive_path, output_dir)


if __name__ == "__main__":
    main()
