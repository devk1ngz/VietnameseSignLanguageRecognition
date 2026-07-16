"""Ghep gloss -> cau tieng Viet qua Qwen3 (Ollama /api/chat).

Prompt + /no_think + strip <think> lay tu recognition/demo_gradio.py (da kiem chung).
"""

import re

import httpx

from serving.config import settings
from serving.services.gloss_decoder import dedupe_consecutive
from serving.utils.logger import get_logger

logger = get_logger(__name__)

# LUU Y: noi dung prompt PHAI viet tieng Viet CO DAU — model doc prompt khong dau
# se sinh cau kem han. (Chi comment trong code moi viet khong dau.)
SYSTEM_PROMPT = (
    "Bạn là trợ lý dịch ngôn ngữ ký hiệu tiếng Việt. Đầu vào là danh sách các từ (gloss) "
    "nhận dạng được từ người ký hiệu, theo đúng thứ tự thể hiện. Nhiệm vụ của bạn: ghép "
    "chúng thành MỘT câu tiếng Việt tự nhiên, đúng ngữ pháp, dễ hiểu.\n"
    "QUY TẮC BẮT BUỘC:\n"
    "- Dùng ĐỦ và ĐÚNG tất cả các từ đã cho, giữ nguyên thứ tự xuất hiện.\n"
    "- KHÔNG thêm từ mang nghĩa mới, KHÔNG bịa nội dung, KHÔNG bỏ bớt từ nào.\n"
    "- Chỉ được thêm tối thiểu hư từ/từ nối (là, và, thì, của, đang, ạ, ơi, không...) "
    "và dấu câu khi thật cần thiết để câu đúng ngữ pháp.\n"
    "- Nếu chỉ có một từ, trả về đúng từ đó (viết hoa chữ đầu, thêm dấu câu).\n"
    "VÍ DỤ:\n"
    "- Từ: tôi, muốn, ăn, cơm → Tôi muốn ăn cơm.\n"
    "- Từ: mẹ, đi, chợ → Mẹ đi chợ.\n"
    "- Từ: anh, khỏe, không → Anh khỏe không?\n"
    "- Từ: xin chào → Xin chào.\n"
    "Chỉ trả về câu kết quả, không giải thích, không dùng thẻ <think> hay định dạng đặc biệt."
)

# Ollama co the cold-start nap model lan dau lau hon timeout thuong.
WARMUP_TIMEOUT = 120.0

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>")
# Qwen3 doi khi lam lo token dieu khien /think, /no_think ra output duoi dang van ban.
_CONTROL_TOKEN_RE = re.compile(r"\s*/(?:no_)?think\b")
# Dong chi chua "nhan" (label) model hay chen truoc cau: "Ban dich", "Cau", "Ket qua"...
_LABEL_ONLY_LINE_RE = re.compile(
    r"^(?:bản dịch|câu(?:\s+dịch|\s+trả lời)?|kết quả|đáp án|trả lời|output|answer)"
    r"\s*[:：.\-–]*\s*$",
    flags=re.IGNORECASE,
)
# Tien to nhan o dau dong, chi bo khi co dau hai cham (tin hieu chac chan la nhan).
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:bản dịch|câu(?:\s+dịch|\s+trả lời)?|kết quả|đáp án|trả lời|output|answer)"
    r"\s*[:：]\s*",
    flags=re.IGNORECASE,
)


def _strip_think(text: str) -> str:
    """Lam sach output LLM: bo khoi/tag suy nghi, token dieu khien va dong nhan thua."""
    text = _THINK_RE.sub("", text)
    text = _THINK_TAG_RE.sub("", text)
    text = _CONTROL_TOKEN_RE.sub("", text)
    cleaned: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _LABEL_ONLY_LINE_RE.match(line):
            continue
        line = _LABEL_PREFIX_RE.sub("", line).strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned).strip()


class LLMService:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    def load(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url, timeout=settings.ollama_timeout
        )

    async def gloss_to_sentence(
        self, gloss_sequence: list[str], timeout=httpx.USE_CLIENT_DEFAULT
    ) -> str:
        gloss_sequence = dedupe_consecutive(gloss_sequence)
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Cac tu nhan dang duoc: {', '.join(gloss_sequence)} /no_think",
                },
            ],
            # Tat reasoning ngay tu API (nhanh nhat + tranh lo <think>); /no_think trong
            # prompt giu lai lam du phong cho ban Ollama cu chua ho tro "think".
            "think": False,
            "options": {
                "temperature": 0.3,
                "num_predict": settings.ollama_num_predict,
            },
            "keep_alive": settings.ollama_keep_alive,
            "stream": False,
        }
        resp = await self._client.post("/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        sentence = _strip_think(content)
        return sentence or " ".join(gloss_sequence)

    async def ping(self) -> None:
        """Nong may (co the cold-start lau) de request that sau do khong bi timeout."""
        await self.gloss_to_sentence(["xin chao"], timeout=WARMUP_TIMEOUT)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
