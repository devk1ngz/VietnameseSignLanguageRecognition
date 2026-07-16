"""Warmup luc khoi dong: nap SPOTER + TTS, va nong may Ollama."""

import asyncio

import numpy as np

from serving.config import settings
from serving.dependencies import (
    get_gloss_decoder,
    get_llm,
    get_slgcn,
    get_spoter,
    get_tts,
    get_video_recognizer,
)
from serving.utils.logger import get_logger

logger = get_logger(__name__)


async def ensure_tts_loaded() -> None:
    """Nap TTS mot lan (idempotent). Dung o warmup va o handler khi can."""
    tts = get_tts()
    if not tts.is_loaded:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, tts.load)


async def warmup_all() -> None:
    loop = asyncio.get_running_loop()

    # Nap dong bo (load() la CPU/GPU-bound) trong executor de khong block event loop.
    await loop.run_in_executor(None, get_gloss_decoder().load)
    await loop.run_in_executor(None, get_spoter().load)
    # SL-GCN (nhanh 2 cua fusion) — chi luong video dung, nhung nap som cho warm.
    try:
        await loop.run_in_executor(None, get_slgcn().load)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nap SL-GCN that bai: %s", exc)

    # VideoRecognizer dung LAI phien ONNX cua SpoterService + SlgcnService -> nap SAU ca hai.
    try:
        await loop.run_in_executor(None, get_video_recognizer().load)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nap VideoRecognizer that bai: %s", exc)

    # Chay thu SPOTER voi input gia de nap graph/CUDA context.
    dummy = np.zeros((settings.spoter_seq_len, 54, 2), dtype=np.float32)
    try:
        await get_spoter().infer(dummy)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warmup SPOTER that bai: %s", exc)

    # Chay thu SL-GCN voi input gia (1, 3, slgcn_seq_len, num_points, 1).
    slgcn = get_slgcn()
    if slgcn.session is not None:
        slgcn_dummy = np.zeros(
            (1, 3, settings.slgcn_seq_len, settings.slgcn_num_points, 1), dtype=np.float32
        )
        try:
            await loop.run_in_executor(None, slgcn.probs, slgcn_dummy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warmup SL-GCN that bai: %s", exc)

    get_llm().load()
    try:
        await get_llm().ping()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama chua san sang (%r). Cau se ghep tu tho khi thieu LLM.", exc)

    # TTS nap model nang tu HuggingFace lan dau; guard de khong lam sap startup.
    try:
        await ensure_tts_loaded()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nap TTS that bai (%s). Se thu lai khi co request TTS.", exc)

    logger.info("Warmup hoan tat.")
