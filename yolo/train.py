"""Train the model."""

from pathlib import Path
import argparse

from ultralytics import YOLO

NUM_EPOCHS = 100
IMAGE_SIZE = 640


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)

    # if the checkpoint exists, resume training
    if checkpoint.exists():
        model_path = checkpoint / "train/weights/last.pt"
        print(f"Resuming training from {model_path.as_posix()}...")
        model = YOLO(model_path)
        model.train(resume=True)
    else:
        # load pretrained model
        print("Starting new training...")
        model = YOLO("models/yolo11n-seg.pt")
        results = model.train(
            data="visual_servor_yolo.yaml",
            epochs=NUM_EPOCHS,
            imgsz=IMAGE_SIZE,
            project=args.checkpoint,
        )


if __name__ == "__main__":
    main()
