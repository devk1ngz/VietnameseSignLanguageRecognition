"""Wrapper inference SPOTER (ONNX).

Client gui keypoint da qua MediaPipe Holistic + chon khop, nen o day chi con:
    (seq_len, 54, 2) pixel -> chuoi chuan hoa -> poses (1, seq_len, 54, 2) -> ONNX -> probabilities.
File spoter_v3.onnx XUAT SAN "probabilities" (softmax trong graph) -> dung THANG, KHONG softmax lai.
Chay trong ThreadPoolExecutor de khong block event loop.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import torch

from serving.config import settings
from serving.recognition_bridge import load_transforms
from serving.utils.logger import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


class SpoterService:
    def __init__(self):
        self.session: ort.InferenceSession | None = None
        self._transforms = None

    def load(self) -> None:
        self._transforms = load_transforms()
        # Giup onnxruntime-gpu tim CUDA/cuDNN cai qua pip (cac goi nvidia-*-cu12).
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                logger.warning("preload_dlls that bai, bo qua.")
        available = ort.get_available_providers()
        providers = [p for p in settings.spoter_providers if p in available]
        self.session = ort.InferenceSession(str(settings.spoter_onnx_path), providers=providers)
        logger.info("SPOTER san sang. ONNX providers: %s", self.session.get_providers())

    def _probs_sync(self, snapshot: np.ndarray) -> np.ndarray:
        """Dong bo. snapshot: (seq_len, 54, 2) pixel -> vector xac suat (num_classes,).

        spoter_v3.onnx xuat san "probabilities" -> dung thang, khong softmax lai.
        """
        tensor = self._transforms(torch.from_numpy(snapshot))
        poses = tensor.unsqueeze(0).cpu().numpy().astype(np.float32)  # (1, seq_len, 54, 2)
        return self.session.run(["probabilities"], {"poses": poses})[0][0]

    def _infer_sync(self, snapshot: np.ndarray) -> tuple[int, float]:
        probs = self._probs_sync(snapshot)
        idx = int(probs.argmax())
        return idx, float(probs[idx])

    def _fused_infer_sync(
        self, spoter_snapshot: np.ndarray, slgcn_snapshot: np.ndarray | None
    ) -> tuple[int, float, bool]:
        """LATE-FUSION SPOTER + SL-GCN cho luong realtime/REST.

        spoter_snapshot: (seq_len, 54, 2). slgcn_snapshot: (T, 27, 3) raw [x,y,conf] hoac None.
        Neu thieu SL-GCN (client cu / khong nap duoc) -> SPOTER-only. Tra (idx, conf, da_fuse).
        """
        p_spoter = self._probs_sync(spoter_snapshot)
        # Import LUOI de tranh vong import voi dependencies.py.
        from serving.dependencies import get_slgcn
        from serving.services.slgcn import preprocess_snapshot

        slgcn = get_slgcn()
        if slgcn_snapshot is not None and slgcn.session is not None:
            p_slgcn = slgcn.probs(preprocess_snapshot(slgcn_snapshot))
            w = settings.fusion_w_slgcn
            probs = w * p_slgcn + (1.0 - w) * p_spoter
            fused = True
        else:
            probs = p_spoter
            fused = False
        idx = int(probs.argmax())
        return idx, float(probs[idx]), fused

    async def infer(self, snapshot: np.ndarray) -> tuple[int, float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._infer_sync, snapshot)

    async def fused_infer(
        self, spoter_snapshot: np.ndarray, slgcn_snapshot: np.ndarray | None
    ) -> tuple[int, float, bool]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, self._fused_infer_sync, spoter_snapshot, slgcn_snapshot
        )
