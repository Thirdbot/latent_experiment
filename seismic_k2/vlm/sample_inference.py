import argparse
from pathlib import Path

from seismic_k2.config import IMAGE_SUFFIXES
from seismic_k2.fault.detection import load_faultnet, run_faultnet


DEFAULT_FAULTNET_WEIGHTS = Path("models/faultnet_yolo/best.pt")
DEFAULT_TEST_ROOT = Path("data/unicamp_namss/test")
DEFAULT_TRAINED_DIR = Path("outputs/k2_attached_vision/final")


def iter_images(root):
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def select_fault_and_nonfault(test_root, faultnet_weights, conf=0.05, max_scan=200):
    faultnet = load_faultnet(faultnet_weights)
    fault_sample = None
    nonfault_sample = None

    for index, image_path in enumerate(iter_images(test_root), start=1):
        if index > max_scan:
            break
        detections = run_faultnet(faultnet, image_path, conf=conf)
        if detections and fault_sample is None:
            fault_sample = (image_path, detections)
        if not detections and nonfault_sample is None:
            nonfault_sample = (image_path, detections)
        if fault_sample is not None and nonfault_sample is not None:
            return fault_sample, nonfault_sample

    return fault_sample, nonfault_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-root",
        default=DEFAULT_TEST_ROOT.as_posix(),
        help="Folder containing test seismic images.",
    )
    parser.add_argument(
        "--faultnet-weights",
        default=DEFAULT_FAULTNET_WEIGHTS.as_posix(),
        help="YOLO fault segmentation weights.",
    )
    parser.add_argument(
        "--faultnet-conf",
        type=float,
        default=0.05,
        help="Fault detector confidence threshold.",
    )
    parser.add_argument(
        "--trained-dir",
        default=DEFAULT_TRAINED_DIR.as_posix(),
        help="Trained K2 attached vision final folder.",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=200,
        help="Maximum number of test images to scan while selecting examples.",
    )
    args = parser.parse_args()

    fault_sample, nonfault_sample = select_fault_and_nonfault(
        test_root=args.test_root,
        faultnet_weights=args.faultnet_weights,
        conf=args.faultnet_conf,
        max_scan=args.max_scan,
    )

    if fault_sample is None:
        raise RuntimeError("Could not find an image with auxiliary YOLO fault detections.")
    if nonfault_sample is None:
        raise RuntimeError("Could not find an image without auxiliary YOLO fault detections.")

    examples = [
        ("auxiliary_fault_detected", fault_sample),
        ("no_auxiliary_fault_detected", nonfault_sample),
    ]

    from seismic_k2.vlm.k2_vision import run_pipeline

    for name, (image_path, detections) in examples:
        print("=" * 80)
        print(f"example: {name}")
        print(f"image: {image_path}")
        print(f"faultnet_detections: {len(detections)}")
        if detections:
            print(f"top_detection: {detections[0]}")
        run_pipeline(
            image_path=image_path,
            trained_dir=Path(args.trained_dir),
            vision_token_drop_rate=0.0,
        )


if __name__ == "__main__":
    main()
