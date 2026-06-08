import json
import os
from pathlib import Path


K2_REPO_ID = os.getenv("K2_REPO_ID", "daven3/k2")
K2_MODEL_DIR = Path(os.getenv("K2_MODEL_DIR", "models/k2"))
K2_REVISION = os.getenv("K2_REVISION") or None


def has_complete_snapshot(model_dir):
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    if not config_path.exists() or not index_path.exists():
        return False
    index = json.loads(index_path.read_text(encoding="utf-8"))
    files = set(index.get("weight_map", {}).values())
    return bool(files) and all((model_dir / file).exists() for file in files)


def download_k2_snapshot(repo_id=K2_REPO_ID, model_dir=K2_MODEL_DIR, revision=K2_REVISION):
    model_dir = Path(model_dir)
    if has_complete_snapshot(model_dir):
        print(f"K2 already exists at {model_dir}")
        return {"repo_id": repo_id, "model_dir": model_dir.as_posix(), "downloaded": False}

    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {model_dir}")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=model_dir.as_posix(),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    result = {
        "repo_id": repo_id,
        "revision": revision or "default",
        "model_dir": model_dir.as_posix(),
        "snapshot_path": snapshot_path,
        "downloaded": True,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    download_k2_snapshot()
