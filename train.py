from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODEL = Path(__file__).parent / "ultralytics/cfg/models/ISD-YOLO/ISD-YOLO.yaml"


def parse_args():
    parser = ArgumentParser(description="Train ISD-YOLO for single-class invoice seal detection.")
    parser.add_argument("--data", required=True, help="Path to an Ultralytics dataset YAML file.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to the ISD-YOLO model YAML file.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Pretrained weights; use an empty value to disable.")
    parser.add_argument("--device", default=None, help="Training device, for example 0 or cpu.")
    parser.add_argument("--project", default="runs/detect", help="Directory for training outputs.")
    parser.add_argument("--name", default="isd-yolo", help="Experiment name.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)
    model.train(
        data=args.data,
        epochs=200,
        batch=32,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        seed=0,
        deterministic=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

