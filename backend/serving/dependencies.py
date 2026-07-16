"""Singleton cho cac service. Khoi tao mot lan, load() goi trong lifespan (warmup)."""

from serving.services.gloss_decoder import GlossDecoder
from serving.services.llm import LLMService
from serving.services.slgcn import SlgcnService
from serving.services.spoter import SpoterService
from serving.services.tts import TTSService
from serving.services.video_recognizer import VideoRecognizerService

_spoter = SpoterService()
_slgcn = SlgcnService()
_gloss_decoder = GlossDecoder()
_llm = LLMService()
_tts = TTSService()
_video_recognizer = VideoRecognizerService()


def get_spoter() -> SpoterService:
    return _spoter


def get_slgcn() -> SlgcnService:
    return _slgcn


def get_gloss_decoder() -> GlossDecoder:
    return _gloss_decoder


def get_llm() -> LLMService:
    return _llm


def get_tts() -> TTSService:
    return _tts


def get_video_recognizer() -> VideoRecognizerService:
    return _video_recognizer
