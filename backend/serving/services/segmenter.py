"""Segment luong keypoint realtime thanh tung clip 1-ky-hieu (thay cua so truot co dinh).

Port tu RealtimeSegmenter cua demo_gradio.py nhung chay tren CHUOI KEYPOINT (khong co frame
anh, khong chay MediaPipe rieng): goc canh tay tinh thang tu toa do khop than co san trong
vector SPOTER 108 so. Tay gio len (goc khuyu < nguong) = dang ky; tay ha xuong = ket thuc ->
tra ve clip tron 1 ky hieu (dai bien thien). Moi nhanh sau do tu pad theo contract rieng
(SPOTER -> 70, SL-GCN -> 150), nen KHONG can co dinh do dai cua so o tang segment.

Vi model la isolated-sign (khong phai CSLR), nguoi ky can HA TAY giua cac tu de cat sach.
"""

from collections import deque

import numpy as np

from serving.models.schemas import NUM_JOINTS, SLGCN_NUM_JOINTS

# Chi so khop than trong layout SPOTER 54-khop (thu tu JointSelect):
#   [nose, neck, rightEye, leftEye, rightEar, leftEar,
#    rightShoulder(6), leftShoulder(7), rightElbow(8), leftElbow(9), rightWrist(10), leftWrist(11)]
_SIDES = {
    "LEFT": (7, 9, 11),   # (shoulder, elbow, wrist)
    "RIGHT": (6, 8, 10),
}


def _arm_angle(shoulder, elbow, wrist) -> float:
    """Goc khuyu tay (vai-khuyu-co tay), do. Tay buong thang ~180, gio/gap lai nho hon."""
    radians = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(
        shoulder[1] - elbow[1], shoulder[0] - elbow[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def _track_motion(track) -> float:
    """Bien do di chuyen lon nhat (pixel) cua chuoi vi tri co tay."""
    pts = np.array([p for p in track if p is not None], dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.ptp(pts, axis=0).max())


class KeypointSegmenter:
    """May trang thai tung-frame: cat luong keypoint thanh clip 1-ky-hieu theo tay len/xuong."""

    def __init__(
        self,
        angle_threshold: float,
        min_up_frames: int,
        min_down_frames: int,
        min_sign_frames: int,
        max_sign_frames: int,
        preroll_frames: int,
        min_wrist_motion_px: float,
    ):
        self.angle_threshold = angle_threshold
        self.min_up_frames = min_up_frames
        self.min_down_frames = min_down_frames
        self.min_sign_frames = min_sign_frames
        self.max_sign_frames = max_sign_frames
        self.min_wrist_motion_px = min_wrist_motion_px
        self._preroll_spoter: deque = deque(maxlen=preroll_frames)
        self._preroll_slgcn: deque = deque(maxlen=preroll_frames)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.up_count = 0
        self.down_count = 0
        self._spoter_buf: list[np.ndarray] = []   # moi phan tu (54, 2)
        self._slgcn_buf: list[np.ndarray | None] = []  # moi phan tu (27, 3) hoac None
        self._wrist_track: list = []
        self._preroll_spoter.clear()
        self._preroll_slgcn.clear()

    def _hands_up(self, spoter_kp: np.ndarray):
        """(tay dang gio?, toa do co tay nhin thay). spoter_kp: (54, 2) pixel."""
        up = False
        wrists = []
        for shoulder_i, elbow_i, wrist_i in _SIDES.values():
            wr = spoter_kp[wrist_i]
            if wr[0] == 0 and wr[1] == 0:  # co tay khong duoc phat hien -> khop = 0
                continue
            wrists.append((float(wr[0]), float(wr[1])))
            if _arm_angle(spoter_kp[shoulder_i], spoter_kp[elbow_i], wr) < self.angle_threshold:
                up = True
        return up, wrists

    def push(
        self, spoter_frame: np.ndarray, slgcn_frame: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Nap 1 frame keypoint. Tra ve (clip SPOTER (T,54,2), clip SL-GCN (T,27,3)|None)
        khi vua ket thuc mot ky hieu, nguoc lai None."""
        sp = np.asarray(spoter_frame, dtype=np.float32).reshape(NUM_JOINTS, 2)
        sg = (
            np.asarray(slgcn_frame, dtype=np.float32).reshape(SLGCN_NUM_JOINTS, 3)
            if slgcn_frame is not None
            else None
        )
        up, wrists = self._hands_up(sp)
        wrist = tuple(np.mean(wrists, axis=0)) if wrists else None

        if not self.active:
            self._preroll_spoter.append(sp)
            self._preroll_slgcn.append(sg)
            if up:
                self.up_count += 1
                if self.up_count >= self.min_up_frames:
                    # Bat dau ky hieu: mo dem bang pre-roll de giu tron phan dau.
                    self.active = True
                    self._spoter_buf = list(self._preroll_spoter)
                    self._slgcn_buf = list(self._preroll_slgcn)
                    self._wrist_track = [wrist]
                    self.down_count = 0
            else:
                self.up_count = 0
            return None

        # Dang trong mot ky hieu.
        self._spoter_buf.append(sp)
        self._slgcn_buf.append(sg)
        self._wrist_track.append(wrist)
        self.down_count = 0 if up else self.down_count + 1

        finished = self.down_count >= self.min_down_frames or len(self._spoter_buf) >= self.max_sign_frames
        if not finished:
            return None

        spoter_buf, slgcn_buf, track = self._spoter_buf, self._slgcn_buf, self._wrist_track
        # Reset ve trang thai nghi, san sang cho ky hieu tiep theo.
        self.active = False
        self.up_count = 0
        self.down_count = 0
        self._spoter_buf = []
        self._slgcn_buf = []
        self._wrist_track = []
        self._preroll_spoter.clear()
        self._preroll_slgcn.clear()

        return self._finalize_clip(spoter_buf, slgcn_buf, track)

    def flush(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Chot clip dang do (neu co) — dung khi ket thuc phien / khi client bao end_sign."""
        if not self.active:
            return None
        spoter_buf, slgcn_buf, track = self._spoter_buf, self._slgcn_buf, self._wrist_track
        self.reset()
        return self._finalize_clip(spoter_buf, slgcn_buf, track)

    def _finalize_clip(self, spoter_buf, slgcn_buf, track):
        if len(spoter_buf) < self.min_sign_frames:
            return None  # qua ngan -> nhieu
        if _track_motion(track) < self.min_wrist_motion_px:
            return None  # tay gan nhu dung yen -> tu the nghi, khong phai ky hieu
        spoter_clip = np.stack(spoter_buf).astype(np.float32)  # (T, 54, 2)
        # Chi fuse khi CO du frame SL-GCN cho toan clip (dong bo thoi gian).
        if all(f is not None for f in slgcn_buf):
            slgcn_clip = np.stack(slgcn_buf).astype(np.float32)  # (T, 27, 3)
        else:
            slgcn_clip = None
        return spoter_clip, slgcn_clip
