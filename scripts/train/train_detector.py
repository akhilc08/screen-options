"""Fine-tune GPA-GUI-Detector (YOLO11m, one class) on AX boxes from the sandbox screens (scripts/data/det_dataset.py).

UI screenshots aren't mirror-symmetric (text, left-aligned sidebars), so horizontal flips are off; mosaic stays on
for most epochs and is closed for the last few, as is usual for fine-tuning. Validation is on the held-out val apps.

  python scripts/train/train_detector.py --data data/det --labels ax --imgsz 1280 --epochs 15 --name axdet_1280
  -> runs/det/<name>/weights/best.pt
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/det")
    ap.add_argument("--labels", default="ax", choices=("ax", "ax+gpa"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = no cap)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="runs/det")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    root = (Path(args.data) / args.labels).resolve()
    yaml = Path(args.data).resolve() / f"{args.labels}.abs.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: element\n")
    model = YOLO(hf_hub_download("Salesforce/GPA-GUI-Detector", "model.pt"))
    model.train(data=str(yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, workers=args.workers,
                optimizer="AdamW", lr0=2e-4, cos_lr=True, warmup_epochs=1, close_mosaic=3,
                fliplr=0.0, mosaic=0.5, scale=0.3, translate=0.05, hsv_h=0.0, single_cls=True,
                time=args.hours or None, project=str(Path(args.project).resolve()), name=args.name,
                exist_ok=True, plots=False, cache=False, patience=0)


if __name__ == "__main__":
    main()
