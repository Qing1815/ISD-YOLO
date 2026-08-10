from argparse import ArgumentParser

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="Evaluate a trained ISD-YOLO checkpoint on a fixed test split.")
    parser.add_argument("--weights", required=True, help="Path to a trained ISD-YOLO checkpoint.")
    parser.add_argument("--data", required=True, help="Path to an Ultralytics dataset YAML file.")
    parser.add_argument("--device", default=None, help="Evaluation device, for example 0 or cpu.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.val(
        data=args.data,
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.70,
        max_det=300,
        device=args.device,
    )


if __name__ == "__main__":
    main()

