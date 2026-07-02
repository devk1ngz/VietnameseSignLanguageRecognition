"""
Visualize MediaPipe Holistic keypoints on a video file or webcam feed.

Usage (from src/):
    python test_holistic.py --source ../data/processed/vsl_400/cam_1/some_video.mp4
    python test_holistic.py                      # webcam
    python test_holistic.py --source 0           # webcam explicit
    python test_holistic.py --source video.mp4 --save output.mp4
    python test_holistic.py --source video.mp4 --save output.mp4 --headless

Press Q to quit (when display is available).
"""

import argparse
import os
import sys
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


LANDMARK_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
HAND_LANDMARK_STYLE = mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=1, circle_radius=2)
HAND_CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1)


def overlay_stats(frame, fps: float, frame_idx: int, detected: bool) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (260, 75), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx}", (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    status = "Detected" if detected else "No person"
    color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(frame, status, (8, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def draw_all_landmarks(frame, results) -> bool:
    detected = results.pose_landmarks is not None

    # Pose body skeleton
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_STYLE,
            connection_drawing_spec=CONNECTION_STYLE,
        )

    # Face mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(200, 200, 200), thickness=1, circle_radius=1
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(150, 150, 150), thickness=1
            ),
        )

    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_LANDMARK_STYLE,
            connection_drawing_spec=HAND_CONNECTION_STYLE,
        )

    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_LANDMARK_STYLE,
            connection_drawing_spec=HAND_CONNECTION_STYLE,
        )

    return detected


def has_display() -> bool:
    if sys.platform == "linux" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        return True
    except Exception:
        return False


def run(source, save_path: str | None, model_complexity: int, confidence: float, headless: bool) -> None:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}", file=sys.stderr)
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_src, (w, h))
        if not writer.isOpened():
            print(f"[ERROR] Could not open video writer for: {save_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Saving output to: {save_path}")

    show = not headless and has_display()
    if not show:
        print("[INFO] No display detected — running headless (save-only).")
        if not save_path:
            print("[WARN] No --save path given and no display; output will be discarded.", file=sys.stderr)
    else:
        print("Press Q to quit.")

    holistic = mp_holistic.Holistic(
        model_complexity=model_complexity,
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence,
    )

    tick = cv2.getTickFrequency()
    frame_idx = 0
    t_prev = cv2.getTickCount()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        detected = draw_all_landmarks(out, results)

        t_now = cv2.getTickCount()
        fps = tick / (t_now - t_prev)
        t_prev = t_now

        overlay_stats(out, fps, frame_idx, detected)
        frame_idx += 1

        if total > 0 and frame_idx % 30 == 0:
            pct = frame_idx / total * 100
            print(f"\r  {frame_idx}/{total} frames ({pct:.0f}%)", end="", flush=True)

        if writer:
            writer.write(out)

        if show:
            cv2.imshow("MediaPipe Holistic — press Q to quit", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    print(f"\nDone. Processed {frame_idx} frames.")
    cap.release()
    holistic.close()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test MediaPipe Holistic on video/webcam")
    parser.add_argument("--source", default="0",
                        help="Video file path or camera index (default: 0 = webcam)")
    parser.add_argument("--save", default=None, metavar="OUTPUT.mp4",
                        help="Optional path to save the annotated video")
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                        help="Holistic model complexity (0=fastest, 2=most accurate)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Min detection/tracking confidence (default: 0.5)")
    parser.add_argument("--headless", action="store_true",
                        help="Force headless mode (no window, save-only)")
    args = parser.parse_args()
    run(args.source, args.save, args.complexity, args.confidence, args.headless)


if __name__ == "__main__":
    main()
