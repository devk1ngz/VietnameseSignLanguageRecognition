"""Nhan dang ky hieu tu FILE video (upload/quay webcam).

Khac voi luong realtime (client gui keypoint qua WebSocket), o day server nhan ca file video
tho, tu doc frame -> tu cat thanh nhieu ky hieu (heuristic goc canh tay) -> phan loai tung doan.
Logic port tu backend/demo_gradio.py (KHONG import demo vi demo co side-effect luc import:
nap ONNX, in log...). Dung LAI phien ONNX da nap cua SpoterService -> chi mot phien duy nhat.

Chay trong ThreadPoolExecutor vi cv2 + MediaPipe deu blocking.
"""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np

from serving.config import settings
from serving.recognition_bridge import (
    load_pose_extract,
    load_slgcn_transforms,
    load_spoter_graph_transforms,
)
from serving.services.gloss_decoder import dedupe_consecutive
from serving.utils.logger import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=1)
_mp_pose = mp.solutions.pose


def _arm_angle(shoulder, elbow, wrist):
    """Goc khuyu tay (vai-khuyu-co tay), do. Tay buong thang ~180, gio/gap lai nho hon."""
    radians = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(
        shoulder[1] - elbow[1], shoulder[0] - elbow[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def _hands_up_from_pose(pose, frame_rgb, angle_threshold, visibility_threshold):
    """(tay dang gio?, danh sach toa do co tay nhin thay [chuan hoa 0..1]).

    "Dang gio" = it nhat mot tay co goc khuyu < nguong + co tay du hien thi.
    """
    res = pose.process(frame_rgb)
    if not res.pose_landmarks:
        return False, []
    lm = res.pose_landmarks.landmark
    P = _mp_pose.PoseLandmark
    up = False
    wrists = []
    for side in ("LEFT", "RIGHT"):
        sh = lm[getattr(P, f"{side}_SHOULDER").value]
        el = lm[getattr(P, f"{side}_ELBOW").value]
        wr = lm[getattr(P, f"{side}_WRIST").value]
        if wr.visibility < visibility_threshold:
            continue
        wrists.append((wr.x, wr.y))
        if _arm_angle((sh.x, sh.y), (el.x, el.y), (wr.x, wr.y)) < angle_threshold:
            up = True
    return up, wrists


def _track_motion(track):
    """Bien do di chuyen lon nhat (toa do chuan hoa) cua chuoi vi tri co tay."""
    pts = np.array([p for p in track if p is not None], dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.ptp(pts, axis=0).max())


def read_frames(path: str):
    """Doc file video -> (frames RGB, fps, width, height). Loi neu khong co frame nao."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError("Khong doc duoc khung hinh nao tu video.")
    h, w = frames[0].shape[:2]
    return frames, float(fps), int(w), int(h)


def segment_video(
    frames,
    angle_threshold=150.0,
    visibility_threshold=0.6,
    min_up_frames=3,
    min_down_frames=8,
    min_sign_frames=8,
    max_sign_frames=90,
    preroll_frames=5,
    min_wrist_motion=0.04,
    model_complexity=None,
):
    """Tach video NHIEU ky hieu thanh danh sach cac doan frame (moi doan ~1 ky hieu).

    Chay offline cung logic tay len/xuong voi mode realtime (MediaPipe Pose tung frame).
    Doan ma co tay gan nhu dung yen (< min_wrist_motion) bi loai — tu the nghi lot qua nguong goc.
    model_complexity: do phuc tap MediaPipe Pose (mac dinh lay tu config, 0 = nhanh nhat).
    """
    if model_complexity is None:
        model_complexity = settings.video_pose_model_complexity
    pose = _mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
        model_complexity=model_complexity,
    )
    segments, active, up, down, buf, track = [], False, 0, 0, [], []
    preroll = deque(maxlen=preroll_frames)
    try:
        for fr in frames:
            hands_up, wrists = _hands_up_from_pose(
                pose, fr, angle_threshold, visibility_threshold
            )
            wrist = tuple(np.mean(wrists, axis=0)) if wrists else None
            if not active:
                preroll.append(fr)
                if hands_up:
                    up += 1
                    if up >= min_up_frames:
                        active, buf, down, track = True, list(preroll), 0, []
                else:
                    up = 0
            else:
                buf.append(fr)
                track.append(wrist)
                down = 0 if hands_up else down + 1
                if down >= min_down_frames or len(buf) >= max_sign_frames:
                    if len(buf) >= min_sign_frames and _track_motion(track) >= min_wrist_motion:
                        segments.append(buf)
                    active, up, down, buf, track = False, 0, 0, [], []
                    preroll.clear()
        # Doan con do o cuoi video (chua kip ha tay).
        if active and len(buf) >= min_sign_frames and _track_motion(track) >= min_wrist_motion:
            segments.append(buf)
    finally:
        pose.close()
    return segments


class VideoRecognizerService:
    def __init__(self):
        self._pose_extract = None
        self._spoter_transforms = None
        self._slgcn_transforms = None

    def load(self) -> None:
        """Nap tien xu ly cho LATE-FUSION: PoseExtract dung chung + nhanh SPOTER + nhanh SL-GCN.

        Phien ONNX dung LAI cua SpoterService + SlgcnService (nap o warmup)."""
        self._pose_extract = load_pose_extract()
        self._spoter_transforms = load_spoter_graph_transforms()
        self._slgcn_transforms = load_slgcn_transforms()
        logger.info(
            "VideoRecognizer san sang — fusion SPOTER + SL-GCN (W_SLGCN=%.2f), "
            "dung lai phien ONNX cua SpoterService/SlgcnService.",
            settings.fusion_w_slgcn,
        )

    def _classify_frames(self, frames, fps, w, h) -> tuple[str | None, float]:
        """Danh sach frame RGB -> (gloss top-1 | None, conf) qua LATE-FUSION SPOTER + SL-GCN.

        Trich Pose 1 lan roi re 2 nhanh; hop nhat o tang xac suat:
            P = W_SLGCN * P_slgcn + (1 - W_SLGCN) * P_spoter.
        Ca hai ONNX xuat san "probabilities" -> KHONG softmax lai.
        """
        # Import LUOI de tranh vong import (dependencies.py import module nay).
        from serving.dependencies import get_gloss_decoder, get_slgcn, get_spoter

        spoter_session = get_spoter().session
        if spoter_session is None:
            raise RuntimeError("SpoterService chua nap — khong co phien ONNX de infer.")
        slgcn = get_slgcn()

        # PoseExtract dung chung (chay 1 lan). Ha do phuc tap Holistic -> nhanh hon.
        # LUU Y: khong cat bot frame TRUOC chuan hoa — chuan hoa dung thong ke toan chuoi.
        holistic_config = {"model_complexity": settings.video_holistic_model_complexity}
        pose = self._pose_extract(
            {"frames": frames, "fps": fps, "width": w, "height": h, "holistic_config": holistic_config}
        )

        # Nhanh SPOTER TRUOC (chi doc pixel tho, khong mutate Pose).
        spoter_tensor = self._spoter_transforms(pose)
        spoter_in = spoter_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)  # (1, 70, 54, 2)
        p_spoter = spoter_session.run(["probabilities"], {"poses": spoter_in})[0][0]

        # Nhanh SL-GCN SAU (SLGCNJointSelect goi normalize_distribution -> mutate Pose).
        # Neu SL-GCN chua nap thi lui ve SPOTER-only (khong lam sap luong video).
        if slgcn.session is not None:
            slgcn_arr = self._slgcn_transforms(pose)                        # (3, 150, 27, 1)
            slgcn_in = slgcn_arr[None].astype(np.float32)                   # (1, 3, 150, 27, 1)
            p_slgcn = slgcn.probs(slgcn_in)
            w_slgcn = settings.fusion_w_slgcn
            probs = w_slgcn * p_slgcn + (1.0 - w_slgcn) * p_spoter
        else:
            logger.warning("SL-GCN chua san sang — luong video tam thoi SPOTER-only.")
            probs = p_spoter

        idx = int(probs.argmax())
        gloss = get_gloss_decoder().label(idx)
        return gloss, float(probs[idx])

    def recognize_video_file(self, path: str) -> list[str]:
        """Dong bo (chay trong thread pool): file video -> danh sach tu (gloss) theo thu tu.

        Tu tach video thanh nhieu ky hieu roi phan loai tung doan; chi giu tu du tin cay.
        """
        if self._pose_extract is None:
            self.load()
        frames, fps, w, h = read_frames(path)
        segments = segment_video(frames)
        glosses: list[str] = []
        skipped = 0
        for seg_frames in segments:
            gloss, conf = self._classify_frames(seg_frames, fps, w, h)
            if gloss is None or conf < settings.video_confidence_threshold:
                skipped += 1
                continue
            glosses.append(gloss)
        # Trong MOT video, tu trung lien ke (cung ky hieu bi cat lam nhieu doan) -> giu 1.
        deduped = dedupe_consecutive(glosses)
        logger.info(
            "Video: %d doan cat, %d tu nhan dang (%d sau khi bo trung lien ke), %d doan bo (duoi %.0f%%).",
            len(segments),
            len(glosses),
            len(deduped),
            skipped,
            settings.video_confidence_threshold * 100,
        )
        return deduped

    async def recognize(self, path: str) -> list[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self.recognize_video_file, path)
