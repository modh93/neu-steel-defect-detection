import argparse
from ultralytics import YOLO


def train(data_config: str, epochs: int, patience: int, imgsz: int, batch: int, run_name: str) -> None:
    model = YOLO("yolov8n.pt")

    model.train(
        data=data_config,
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project="runs",
        name=run_name,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement YOLOv8 sur NEU-DET")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Chemin vers data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--run-name", type=str, default="neu_run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_config=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.run_name,
    )