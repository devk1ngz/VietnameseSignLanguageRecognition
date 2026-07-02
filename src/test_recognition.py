"""
Test sign language recognition on a video containing one or more signs.

Two segmentation modes:
  --mode arm      Elbow-angle segmentation — signer drops arms between words (default)
  --mode sliding  Sliding-window — signs are performed continuously back-to-back

Usage (from src/):
    # Continuous signing (option B)
    python test_recognition.py \\
        --arch spoter \\
        --pretrained experiments/spoter_v3.0_multicam/checkpoint-75264 \\
        --source test/my_video.mp4 \\
        --save test_output/result.mp4 \\
        --mode sliding

    # Adjust window / stride (frames)
    python test_recognition.py ... --mode sliding --window 60 --stride 15

Press Q to quit when a display is available.
"""

import argparse
import collections
import csv
import os
import sys
from time import time

import cv2
import numpy as np
import mediapipe as mp

from configs import ModelConfig, InferenceConfig
from data import Arm, get_sample_timestamp, ok_to_get_frame
from tools import load_pipeline
from utils import POSE_BASED_MODELS
from visualization import draw_text_on_image

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

_POSE_DOT  = mp_drawing.DrawingSpec(color=(0, 255, 0),   thickness=1, circle_radius=2)
_POSE_LINE = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1)
_HAND_DOT  = mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=1, circle_radius=3)
_HAND_LINE = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1)
_FACE_DOT  = mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1, circle_radius=1)
_FACE_LINE = mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1)


# ── shared helpers ────────────────────────────────────────────────────────────

def has_display() -> bool:
    if sys.platform == "linux" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except Exception:
        return False


def draw_landmarks(frame: np.ndarray, results) -> None:
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS, _FACE_DOT, _FACE_LINE)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS, _POSE_DOT, _POSE_LINE)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS, _HAND_DOT, _HAND_LINE)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS, _HAND_DOT, _HAND_LINE)


def run_inference(sign_pipeline, frames: list, fps: float, W: int, H: int,
                  use_pose_model: bool, holistic_config: dict = None) -> list:
    if not frames:
        return []
    if use_pose_model:
        # pose_format.load_holistic feeds frames straight to MediaPipe, which
        # requires RGB.  Frames buffered from cv2.VideoCapture are BGR.
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        pipeline_input = {"frames": rgb_frames, "fps": fps, "width": W, "height": H}
        if holistic_config:
            pipeline_input["holistic_config"] = holistic_config
    else:
        pipeline_input = np.array(frames)
    return sign_pipeline(pipeline_input)


def label_key(preds: list) -> str:
    return "gloss" if preds and "gloss" in preds[0] else "label"


def pred_str(preds: list) -> str:
    key = label_key(preds)
    return ", ".join(f"{p[key]} ({p['score']*100:.1f}%)" for p in preds)


def draw_pred_banner(frame: np.ndarray, text: str) -> np.ndarray:
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 50), (frame.shape[1], h), (0, 0, 0), -1)
    return draw_text_on_image(frame, text, (8, h - 42), color=(0, 200, 255), font_size=22)


def draw_log_panel(frame: np.ndarray, pred_log: list[str]) -> np.ndarray:
    if not pred_log:
        return frame
    w = frame.shape[1]
    log_x = w - 400
    rows = pred_log[-12:]
    panel_h = len(rows) * 24 + 30
    cv2.rectangle(frame, (log_x, 0), (w, panel_h), (20, 20, 20), -1)
    frame = draw_text_on_image(frame, "Sign log:", (log_x + 6, 4),
                               color=(255, 255, 100), font_size=16)
    for i, entry in enumerate(rows):
        frame = draw_text_on_image(frame, entry, (log_x + 6, 26 + i * 24),
                                   color=(220, 220, 220), font_size=15)
    return frame


def save_csv(path: str, results_log: list[dict]) -> None:
    if not results_log:
        return
    csv_path = os.path.splitext(path)[0] + "_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results_log[0].keys())
        w.writeheader()
        w.writerows(results_log)
    print(f"Predictions saved to: {csv_path}")


def open_video_writer(path: str, fps: float, W: int, H: int):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if not writer.isOpened():
        print(f"[ERROR] Cannot open writer: {path}", file=sys.stderr)
        sys.exit(1)
    return writer


# ── mode: sliding window ──────────────────────────────────────────────────────

def run_sliding(args, sign_pipeline, cap, fps_src, W, H, writer, show, holistic):
    """
    Divide the video into overlapping windows of fixed length and run
    inference every `stride` frames on the last `window` frames.

    Temporal voting (debounce) collapses the noisy per-window stream into
    discrete signs:  a sign is *confirmed* only when the same top-1 label
    stays on top for `--min_agree` consecutive windows, each scoring at
    least `--min_score`.  A confidence gap (a window below --min_score)
    clears the running candidate, so the next stable sign can re-fire.
    Consecutive confirmations of the same label are de-duplicated.
    """
    window_size = args.window
    stride      = args.stride
    min_score   = args.min_score
    min_agree   = args.min_agree

    ring = collections.deque(maxlen=window_size)
    results_log    = []
    pred_log       = []
    live_text      = ""        # latest raw per-window guess (HUD only)
    confirmed_text = ""        # latest confirmed sign (banner)

    candidate       = None     # label currently accumulating agreement
    candidate_count = 0        # how many consecutive windows agreed
    candidate_preds = None     # full top-k of the agreeing window (for logging)
    cand_start_frame = 0       # frame where the candidate first appeared
    last_confirmed  = None     # last confirmed label (consecutive dedup)
    sign_idx        = 0

    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tick      = cv2.getTickFrequency()
    t_prev    = cv2.getTickCount()
    frame_idx = 0
    frames_since_infer = 0

    print(f"[sliding]  window={window_size}f ({window_size/fps_src:.1f}s)  "
          f"stride={stride}f ({stride/fps_src:.1f}s)  "
          f"min_score={min_score*100:.0f}%  min_agree={min_agree}")
    print("Processing video...")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_results = holistic.process(rgb)
        rgb.flags.writeable = True
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        draw_landmarks(out, mp_results)
        ring.append(frame.copy())
        frames_since_infer += 1

        # ── run inference every `stride` frames once window is full ───────────
        if len(ring) == window_size and frames_since_infer >= stride:
            frames_since_infer = 0
            t0 = time()
            preds = run_inference(sign_pipeline, list(ring), fps_src, W, H,
                                  args.arch in POSE_BASED_MODELS,
                                  getattr(args, "holistic_config", None))
            elapsed = time() - t0

            if preds:
                key   = label_key(preds)
                lbl   = preds[0][key]
                score = float(preds[0]["score"])
                live_text = f"{lbl} ({score*100:.0f}%)"

                # ── temporal voting ──────────────────────────────────────────
                if score < min_score:
                    # confidence gap → break the run, allow re-firing later
                    candidate, candidate_count = None, 0
                    last_confirmed = None
                elif lbl == candidate:
                    candidate_count += 1
                else:
                    candidate, candidate_count = lbl, 1
                    candidate_preds = preds
                    cand_start_frame = max(0, frame_idx - window_size)

                # confirm once the candidate is stable enough
                if (candidate is not None
                        and candidate_count == min_agree
                        and candidate != last_confirmed):
                    sign_idx += 1
                    last_confirmed = candidate
                    ps = pred_str(candidate_preds)
                    t_start = cand_start_frame / fps_src
                    t_end   = frame_idx / fps_src
                    confirmed_text = f"#{sign_idx} {candidate} ({score*100:.0f}%)"
                    pred_log.append(f"#{sign_idx} [{t_start:.1f}s] {ps}")
                    results_log.append({
                        "sign_idx":    sign_idx,
                        "start_s":     f"{t_start:.2f}",
                        "end_s":       f"{t_end:.2f}",
                        "inference_s": f"{elapsed:.3f}",
                        "predictions": ps,
                    })
                    print(f"\n[Sign #{sign_idx}] {t_start:.1f}s–{t_end:.1f}s "
                          f"({elapsed:.2f}s) | {ps}")

        # ── HUD ───────────────────────────────────────────────────────────────
        t_now = cv2.getTickCount()
        fps   = tick / (t_now - t_prev)
        t_prev = t_now

        bar_text = (f"FPS:{fps:.0f}  frame:{frame_idx}  "
                    f"win:{len(ring)}/{window_size}  "
                    f"vote:{candidate_count}/{min_agree}")
        cv2.rectangle(out, (0, 0), (560, 26), (0, 0, 0), -1)
        out = draw_text_on_image(out, bar_text, (6, 3), color=(200, 200, 200), font_size=16)
        if live_text:
            cv2.rectangle(out, (0, 26), (360, 52), (0, 0, 0), -1)
            out = draw_text_on_image(out, f"live: {live_text}", (6, 29),
                                     color=(160, 160, 160), font_size=15)

        if confirmed_text:
            out = draw_pred_banner(out, f"Confirmed: {confirmed_text}")
        out = draw_log_panel(out, pred_log)

        frame_idx += 1
        if total > 0 and frame_idx % 60 == 0:
            print(f"\r  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)", end="", flush=True)

        if writer:
            writer.write(out)
        if show:
            cv2.imshow("Sign Recognition (sliding) — Q to quit", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    print(f"\nDone. {frame_idx} frames processed, {sign_idx} signs confirmed.")
    return results_log


# ── mode: arm-angle segmentation ─────────────────────────────────────────────

def run_arm(args, sign_pipeline, cap, fps_src, W, H, writer, show, holistic):
    """
    Buffer frames while at least one arm is raised above the elbow-angle
    threshold.  Emit a prediction when both arms return to rest.
    """
    left_arm  = Arm("left",  args.visibility)
    right_arm = Arm("right", args.visibility)

    frames_all  = []        # every frame (for contiguous slicing)
    seg_start   = None      # index in frames_all where current sign began
    results_log = []
    pred_log    = []
    last_pred_text = ""
    sign_idx    = 0
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tick        = cv2.getTickFrequency()
    t_prev      = cv2.getTickCount()
    frame_idx   = 0
    use_pose    = args.arch in POSE_BASED_MODELS
    # Pre-roll: recover the sign onset that arm-up detection lag skips
    preroll     = args.min_up_frames + int(args.delay / 1000 * fps_src)

    def flush(clip, start_ms, end_ms):
        nonlocal sign_idx, last_pred_text
        if not clip:
            return
        sign_idx += 1
        t0    = time()
        preds = run_inference(sign_pipeline, clip, fps_src, W, H, use_pose,
                              getattr(args, "holistic_config", None))
        elapsed = time() - t0
        key   = label_key(preds)
        ps    = pred_str(preds)
        last_pred_text = f"#{sign_idx} {preds[0][key]} ({preds[0]['score']*100:.1f}%)"
        pred_log.append(f"#{sign_idx}: {ps}")
        results_log.append({
            "sign_idx": sign_idx, "start_s": f"{start_ms/1000:.2f}",
            "end_s": f"{end_ms/1000:.2f}", "inference_s": f"{elapsed:.3f}",
            "predictions": ps,
        })
        print(f"\n[Sign #{sign_idx}] {start_ms/1000:.2f}s–{end_ms/1000:.2f}s "
              f"({elapsed:.2f}s) | {ps}")

    print("[arm]  angle_threshold={} min_up={} min_down={} visibility={}".format(
        args.angle_threshold, args.min_up_frames, args.min_down_frames, args.visibility))
    print("Processing video...")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_results = holistic.process(rgb)
        rgb.flags.writeable = True
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        draw_landmarks(out, mp_results)

        try:
            lm = mp_results.pose_landmarks.landmark
            left_arm.set_pose(lm)
            right_arm.set_pose(lm)
        except Exception:
            pass

        ms        = cap.get(cv2.CAP_PROP_POS_MSEC)
        left_ok   = ok_to_get_frame(left_arm,  args.angle_threshold,
                                    args.min_up_frames, args.min_down_frames, ms, args.delay)
        right_ok  = ok_to_get_frame(right_arm, args.angle_threshold,
                                    args.min_up_frames, args.min_down_frames, ms, args.delay)
        frames_all.append(frame.copy())
        collecting = left_ok or right_ok
        if collecting and seg_start is None:
            # mark sign onset, backing up by the detection-lag pre-roll
            seg_start = max(0, len(frames_all) - 1 - preroll)

        start_time, end_time = get_sample_timestamp(left_arm, right_arm)
        if start_time != 0 and end_time != 0 and seg_start is not None:
            s = (left_arm.start_time or 0) or (right_arm.start_time or 0)
            e = max(left_arm.end_time or 0, right_arm.end_time or 0)
            # contiguous slice from onset → now (includes onset + any brief gaps)
            clip = frames_all[seg_start:len(frames_all)]
            flush(clip, s, e)
            left_arm.reset_state()
            right_arm.reset_state()
            seg_start = None

        # ── HUD ───────────────────────────────────────────────────────────────
        t_now  = cv2.getTickCount()
        fps    = tick / (t_now - t_prev)
        t_prev = t_now

        cv2.rectangle(out, (0, 0), (340, 90), (0, 0, 0), -1)
        out = draw_text_on_image(out, f"FPS:{fps:.0f}  frame:{frame_idx}", (6, 3),
                                 color=(200, 200, 200), font_size=16)
        out = draw_text_on_image(out,
            f"L:{left_arm.angle:.0f}°  R:{right_arm.angle:.0f}°", (6, 26),
            color=(150, 220, 150), font_size=16)
        nbuf   = (len(frames_all) - seg_start) if seg_start is not None else 0
        status = f"Buffering ({nbuf} frames)" if collecting else "Waiting..."
        color  = (100, 255, 100) if collecting else (180, 180, 180)
        cv2.rectangle(out, (0, 50), (340, 90), (0, 80, 0) if collecting else (0, 0, 0), -1)
        out = draw_text_on_image(out, status, (6, 52), color=color, font_size=18)

        if last_pred_text:
            out = draw_pred_banner(out, f"Prediction: {last_pred_text}")
        out = draw_log_panel(out, pred_log)

        frame_idx += 1
        if total > 0 and frame_idx % 60 == 0:
            print(f"\r  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)", end="", flush=True)

        if writer:
            writer.write(out)
        if show:
            cv2.imshow("Sign Recognition (arm) — Q to quit", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # flush remaining (video ended mid-sign)
    if seg_start is not None:
        s = (left_arm.start_time or 0) or (right_arm.start_time or 0)
        flush(frames_all[seg_start:len(frames_all)], s, frame_idx / fps_src * 1000)

    print(f"\nDone. {frame_idx} frames processed, {sign_idx} signs detected.")
    return results_log


# ── entry point ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # Higher-quality pose extraction for difficult videos (poor hand detection)
    if getattr(args, "hires_pose", False):
        args.holistic_config = {
            "model_complexity": 2,
            "min_detection_confidence": 0.3,
            "min_tracking_confidence": 0.3,
        }
        print(f"[hires_pose] holistic extraction override: {args.holistic_config}")
    else:
        args.holistic_config = None

    model_cfg = ModelConfig(arch=args.arch, pretrained=args.pretrained)
    inf_cfg   = InferenceConfig(
        source=args.source, use_onnx=False, device=args.device,
        cache_dir="models/huggingface", visibility=getattr(args, "visibility", 0.5),
        angle_threshold=getattr(args, "angle_threshold", 140),
        min_num_up_frames=getattr(args, "min_up_frames", 10),
        min_num_down_frames=getattr(args, "min_down_frames", 10),
        delay=getattr(args, "delay", 400), top_k=args.top_k,
    )

    if args.flip:
        print("[flip] Horizontally flipping frames (webcam mirror correction). "
              "Use --no-flip if your video is NOT mirrored.")
    else:
        print("[flip] Disabled — frames used as-is.")

    print(f"Loading pipeline: arch={args.arch}  pretrained={args.pretrained}")
    sign_pipeline = load_pipeline(model_cfg, inf_cfg)
    print("Pipeline ready.")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.source}", file=sys.stderr)
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = open_video_writer(args.save, fps_src, W, H) if args.save else None
    if args.save:
        print(f"Saving to: {args.save}")

    show = not args.headless and has_display()
    if not show:
        print("[INFO] Headless mode — no window will open.")

    holistic = mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if args.mode == "sliding":
        results_log = run_sliding(args, sign_pipeline, cap, fps_src, W, H, writer, show, holistic)
    else:
        results_log = run_arm(args, sign_pipeline, cap, fps_src, W, H, writer, show, holistic)

    cap.release()
    holistic.close()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    if args.save and results_log:
        save_csv(args.save, results_log)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test sign language recognition on a video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # model
    parser.add_argument("--arch",       required=True,
                        help="Model arch: spoter | sl_gcn | swin3d_t | ...")
    parser.add_argument("--pretrained", required=True,
                        help="Local checkpoint path or HuggingFace repo ID")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--top_k",      type=int, default=3)

    # video I/O
    parser.add_argument("--source",   required=True)
    parser.add_argument("--save",     default=None, metavar="OUTPUT.mp4")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True,
                        help="Horizontally flip each frame; ON by default because webcam "
                             "recordings are mirrored. Use --no-flip for un-mirrored video.")

    # segmentation mode
    parser.add_argument("--mode", default="arm", choices=["arm", "sliding"],
                        help="arm: elbow-angle segmentation  |  sliding: fixed-window")
    parser.add_argument("--hires_pose", action="store_true",
                        help="Use MediaPipe model_complexity=2 + lower thresholds for "
                             "better hand detection on difficult videos (pose models only)")

    # sliding-window params
    parser.add_argument("--window",    type=int,   default=90,
                        help="[sliding] frames per inference window; set near your sign length "
                             "(default 90 = 3s @ 30fps)")
    parser.add_argument("--stride",    type=int,   default=15,
                        help="[sliding] frames between inferences    (default 15 = 0.5s @ 30fps)")
    parser.add_argument("--min_score", type=float, default=0.5,
                        help="[sliding] min confidence for a window to count as a vote (0–1)")
    parser.add_argument("--min_agree", type=int,   default=3,
                        help="[sliding] consecutive agreeing windows needed to confirm a sign")

    # arm-mode params
    parser.add_argument("--angle_threshold", type=int,   default=140)
    parser.add_argument("--min_up_frames",   type=int,   default=10)
    parser.add_argument("--min_down_frames", type=int,   default=10)
    parser.add_argument("--delay",           type=int,   default=400)
    parser.add_argument("--visibility",      type=float, default=0.2)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
