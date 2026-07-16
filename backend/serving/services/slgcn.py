"""Wrapper inference SL-GCN ensemble (ONNX) — nhanh thu 2 cua late-fusion.

Input la mang da qua tien xu ly SL-GCN: (1, 3, slgcn_seq_len, num_points, 1); ONNX xuat san
"probabilities" (bone/motion/normalize/softmax-avg 4 stream deu bake trong graph) -> dung THANG.

Hai duong vao san sinh mang do:
  - Luong VIDEO: Pose day du -> SLGCNJointSelect + SLGCNPad (trong recognition_bridge).
  - Luong REALTIME/REST: client gui san keypoint tho (x, y, confidence) cho 27 joint;
    `preprocess_snapshot()` o day tai hien CHINH XAC buoc normalize_distribution + pad
    ma AI core lam tren Pose (da kiem chung bit-exact, diff = 0):
      * normalize x,y theo TUNG joint: (v - mean)/std tren cac frame CO confidence>0;
        frame thieu (confidence==0) giu 0 (dung ngu nghia masked-array cua pose_format).
      * kenh confidence giu nguyen.
      * pad/lap ve slgcn_seq_len roi doi truc -> (3, T, 27, 1).
Chay dong bo (goi trong thread pool).
"""

import numpy as np
import onnxruntime as ort

from serving.config import settings
from serving.utils.logger import get_logger

logger = get_logger(__name__)


def _pad(data: np.ndarray, num_frames: int) -> np.ndarray:
    """(T, V, 3) -> (3, num_frames, V, 1). Lap chuoi neu ngan, cat neu dai (khop SLGCNPad)."""
    V, C = data.shape[1], data.shape[2]
    padded = np.zeros((num_frames, V, C, 1), dtype=np.float32)
    L = data.shape[0]
    if L == 0:
        return np.transpose(padded, [2, 0, 1, 3])
    if L < num_frames:
        padded[:L, :, :, 0] = data
        rest = num_frames - L
        num = int(np.ceil(rest / L))
        pad = np.concatenate([data for _ in range(num)], 0)[:rest]
        padded[L:, :, :, 0] = pad
    else:
        padded[:, :, :, 0] = data[:num_frames, :, :]
    return np.transpose(padded, [2, 0, 1, 3])


def preprocess_snapshot(snapshot: np.ndarray) -> np.ndarray:
    """(T, 27, 3) raw [x, y, confidence] -> (1, 3, slgcn_seq_len, 27, 1) float32.

    Tai hien normalize_distribution (per-joint, tren frame co confidence>0; frame thieu = 0)
    + pad. Da kiem chung khop TUYET DOI voi AI core tren video that.
    """
    snap = np.asarray(snapshot, dtype=np.float64)
    T, V, _ = snap.shape
    norm = np.zeros((T, V, 3), dtype=np.float64)
    conf = snap[:, :, 2]
    valid = conf > 0
    for ch in (0, 1):
        for k in range(V):
            m = valid[:, k]
            vals = snap[m, k, ch]
            if vals.size:
                std = vals.std()
                if std > 0:
                    z = (snap[:, k, ch] - vals.mean()) / std
                    norm[m, k, ch] = z[m]  # chi frame co confidence; frame thieu giu 0
    norm[:, :, 2] = conf
    return _pad(norm, settings.slgcn_seq_len)[None].astype(np.float32)


class SlgcnService:
    def __init__(self):
        self.session: ort.InferenceSession | None = None

    def load(self) -> None:
        # Giup onnxruntime-gpu tim CUDA/cuDNN cai qua pip (cac goi nvidia-*-cu12).
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                logger.warning("preload_dlls that bai, bo qua.")
        available = ort.get_available_providers()
        providers = [p for p in settings.spoter_providers if p in available]
        self.session = ort.InferenceSession(str(settings.slgcn_onnx_path), providers=providers)
        logger.info("SL-GCN san sang. ONNX providers: %s", self.session.get_providers())

    def probs(self, slgcn_input: np.ndarray) -> np.ndarray:
        """slgcn_input: (1, 3, slgcn_seq_len, num_points, 1) float32 -> probabilities (num_classes,)."""
        if self.session is None:
            raise RuntimeError("SlgcnService chua nap — khong co phien ONNX de infer.")
        return self.session.run(["probabilities"], {"poses": slgcn_input})[0][0]
