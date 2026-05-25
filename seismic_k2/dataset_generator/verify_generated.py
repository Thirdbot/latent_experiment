from __future__ import annotations

import argparse
import json
from pathlib import Path

from seismic_k2.config import GENERATED_DATA_DIR
from seismic_k2.dataset_generator.nli import verify_generated_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify generated seismic instruction JSONL with NLI checks.")
    parser.add_argument(
        "--input",
        default=(GENERATED_DATA_DIR / "train.jsonl").as_posix(),
        help="Generated JSONL file to verify.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Verified JSONL path. Defaults to <input stem>_verified.jsonl.",
    )
    parser.add_argument(
        "--filtered-output",
        default=None,
        help="Filtered JSONL path. Defaults to <input stem>_filtered.jsonl.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional local seismic NLI verifier model. If omitted, uses deterministic fault-evidence rules.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_verified.jsonl")
    filtered_path = (
        Path(args.filtered_output)
        if args.filtered_output
        else input_path.with_name(f"{input_path.stem}_filtered.jsonl")
    )
    summary = verify_generated_dataset(
        input_path=input_path,
        output_path=output_path,
        filtered_output_path=filtered_path,
        model_path=Path(args.model) if args.model else None,
        max_length=args.max_length,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
