
from multiprocessing import freeze_support

import torch
from ultralytics import YOLO


def main():
    print("CUDA available:", torch.cuda.is_available())

    model = YOLO("yolo11s-seg.pt")

    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=640,
        batch=4,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,  # Simplest option for Windows debugging
        project="runs",
        name="cervical_test",
    )


if __name__ == "__main__":
    freeze_support()
    main()
