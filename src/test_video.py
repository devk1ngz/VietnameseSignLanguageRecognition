"""Classify a single pre-segmented sign clip with a trained pose model.

Feeds the whole clip to the model once (the mode matching how the model was
trained / evaluated). For continuous webcam-style streams that need arms up/down
segmentation, use inference.py instead.

Usage (from src/):
    python test_video.py test/cam-on.mp4
    python test_video.py test/cam-on.mp4 --top_k 10
    python test_video.py /path/to/clip.mp4 \
        --pretrained experiments/spoter_v3.0_multicam/checkpoint-75264 \
        --arch spoter --device cuda
"""
import sys
import argparse

import cv2
import numpy as np

from configs import ModelConfig
from tools.models import register_pipeline

DEFAULT_CKPT = "experiments/spoter_v3.0_multicam/checkpoint-75264"


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="path to a video file (.mp4, .avi, ...)")
    p.add_argument("--pretrained", default=DEFAULT_CKPT,
                   help=f"model path or HF repo id (default: {DEFAULT_CKPT})")
    p.add_argument("--arch", default="spoter",
                   help="pose model architecture: spoter | sl_gcn | dsta_slr (default: spoter)")
    p.add_argument("--top_k", type=int, default=5, help="number of predictions to show (default: 5)")
    p.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    p.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True,
                   help="horizontally flip each frame; ON by default because webcam "
                        "recordings are mirrored. Use --no-flip for un-mirrored video.")
    return p.parse_args()


def read_frames(path, flip=True):
    """Read a video into an (T, H, W, C) RGB array plus fps/width/height."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Error: could not open video '{path}'")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        sys.exit(f"Error: no frames decoded from '{path}'")
    frames = np.array(frames)
    h, w = frames.shape[1:3]
    return frames, float(fps), int(w), int(h)


def main():
    args = get_args()

    # Build the pose pipeline (MediaPipe Holistic runs inside preprocess()).
    model_config = ModelConfig(arch=args.arch, pretrained=args.pretrained)
    pipe = register_pipeline(model_config)
    if args.device != "cpu":
        pipe.model.to(args.device)
        pipe.device = pipe.model.device

    if args.flip:
        print("[flip] Horizontally flipping frames (webcam mirror correction). "
              "Use --no-flip if your video is NOT mirrored.")
    else:
        print("[flip] Disabled — frames used as-is.")

    frames, fps, w, h = read_frames(args.video, flip=args.flip)
    preds = pipe({"frames": frames, "fps": fps, "width": w, "height": h}, top_k=args.top_k)

    print(f"\nVideo : {args.video}  ({len(frames)} frames @ {fps:g} fps, {w}x{h})")
    print(f"Model : {args.pretrained}")
    print("-" * 50)
    for rank, pred in enumerate(preds, 1):
        print(f"  {rank:>2}. {pred['gloss']:<28} {float(pred['score']) * 100:6.2f}%")
    print("-" * 50)
    print(f"Prediction: {preds[0]['gloss']}  ({float(preds[0]['score']) * 100:.2f}%)\n")


if __name__ == "__main__":
    main()
