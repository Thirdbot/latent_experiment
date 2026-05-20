import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_DATA_YAML = Path("data/fault_yolo/data.yaml")
DEFAULT_BASE_MODEL = Path("models/fault_yolo/yolo11x-seg.pt")
DEFAULT_OUTPUT_PROJECT = Path("outputs/fault_yolo")


def train_yolo_fault(
    data_yaml=DEFAULT_DATA_YAML,
    model_path=DEFAULT_BASE_MODEL,
    output_project=DEFAULT_OUTPUT_PROJECT,
    run_name="train",
    epochs=100,
    imgsz=640,
    batch=4,
    resume=False,
):
    model = YOLO(Path(model_path).as_posix())
    return model.train(
        data=Path(data_yaml).as_posix(),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=Path(output_project).as_posix(),
        name=run_name,
        task="segment",
        resume=resume,
    )


def evaluate_yolo_fault(weights_path, data_yaml=DEFAULT_DATA_YAML, split="test", imgsz=640):
    model = YOLO(Path(weights_path).as_posix())
    return model.val(
        data=Path(data_yaml).as_posix(),
        split=split,
        imgsz=imgsz,
        task="segment",
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-yaml", default=DEFAULT_DATA_YAML.as_posix())
    train_parser.add_argument("--model", default=DEFAULT_BASE_MODEL.as_posix())
    train_parser.add_argument("--output-project", default=DEFAULT_OUTPUT_PROJECT.as_posix())
    train_parser.add_argument("--run-name", default="train")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=4)
    train_parser.add_argument("--resume", action="store_true")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--weights", required=True)
    eval_parser.add_argument("--data-yaml", default=DEFAULT_DATA_YAML.as_posix())
    eval_parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    eval_parser.add_argument("--imgsz", type=int, default=640)

    args = parser.parse_args()
    if args.command == "train":
        train_yolo_fault(
            data_yaml=args.data_yaml,
            model_path=args.model,
            output_project=args.output_project,
            run_name=args.run_name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            resume=args.resume,
        )
    elif args.command == "eval":
        evaluate_yolo_fault(
            weights_path=args.weights,
            data_yaml=args.data_yaml,
            split=args.split,
            imgsz=args.imgsz,
        )


if __name__ == "__main__":
    main()
