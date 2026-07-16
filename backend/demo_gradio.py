"""
backend/demo_gradio.py

Demo Gradio trọn gói — KHÔNG cần frontend/serving/WebSocket:
    video ký hiệu -> nhận dạng gloss (FUSION SPOTER + SL-GCN, ONNX) -> ghép câu (Qwen3/Ollama) -> giọng nói (VieNeu-TTS)

Nhận dạng là LATE-FUSION 2 model ONNX (xem docs/local_fusion_deployment.md): trích Pose
MediaPipe Holistic 1 lần, rẽ 2 nhánh tiền xử lý (SPOTER 70 frame / SL-GCN 150 frame),
hợp nhất ở tầng xác suất P_fused = 0.75*P_slgcn + 0.25*P_spoter rồi argmax.
LLM và TTS nạp LƯỜI (chỉ khi dùng) và bắt lỗi an toàn để demo không sập khi thiếu Ollama.

Chạy (từ thư mục backend/):
    uv run python demo_gradio.py
Rồi mở http://localhost:7860
"""

import os
import re
import sys
import csv
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

import onnxruntime as ort
import gradio as gr
import mediapipe as mp
from torchvision.transforms.v2 import Compose

from pose_format import Pose

from pipelines.spoter_graph_classification import (
    PoseExtract,
    JointSelect,
    TensorToDict,
    SingleBodyDictNormalize,
    SPOTERSingleHandDictNormalize,
    DictToTensor,
    Shift,
    Pad,
)
from utils import SLGCN_JOINTS, COCO_TO_POSE_FORMAT

# --- Cấu hình ---
# Late-fusion 2 mô hình (xem docs/local_fusion_deployment.md):
#   P_fused = W_SLGCN * P_slgcn + (1 - W_SLGCN) * P_spoter  ->  argmax -> gloss
# Cả hai ONNX đều XUẤT SẴN "probabilities" (đã softmax trong graph) -> KHÔNG softmax lại.
SPOTER_ONNX_PATH = ROOT / "models" / "spoter" / "spoter_v3.onnx"
SLGCN_ONNX_PATH = ROOT / "models" / "sl-gcn" / "sl_gcn_ensemble.onnx"
GLOSS_CSV = ROOT / "experiments" / "gloss.csv"
# Trọng số đã chốt trên validation/test: SL-GCN 0.75, SPOTER 0.25 (fusion 90.1% > mỗi model đơn).
W_SLGCN = float(os.getenv("FUSION_W_SLGCN", "0.75"))
SPOTER_NUM_FRAMES = 70    # cửa sổ của SPOTER
SLGCN_NUM_FRAMES = 150    # cửa sổ của SL-GCN
SLGCN_NUM_POINTS = 27     # số joint của SL-GCN
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
# Ngưỡng xác suất top-1 để TỰ ĐỘNG chấp nhận một từ (realtime / video nhiều từ).
# Model 400 lớp luôn trả về MỘT lớp kể cả với đoạn rác, nên phải lọc theo confidence.
MIN_CONFIDENCE = float(os.getenv("SIGN_MIN_CONF", "0.5"))


# --- Tiền xử lý nhánh SL-GCN ---
# Copy nguyên hai class từ src/features/transforms/sl_gcn.py (dòng 30–73), CỐ TÌNH bỏ
# import `_build_edges` — bone/motion/normalize đã bake vào đồ thị ONNX nên local không cần.

class SLGCNJointSelect:
    """Chọn 27 joint cho SL-GCN, mỗi joint gồm (x, y, confidence)."""

    def __init__(self, num_points: int = SLGCN_NUM_POINTS) -> None:
        self.joints = SLGCN_JOINTS[num_points]

    def __get_point(self, component: str, point: str, pose: Pose) -> np.ndarray:
        idx = pose.header._get_point_index(component, point)
        T, _, _, C = pose.body.data.shape
        data = np.zeros((T, C), dtype=pose.body.data.dtype)
        data[:, :2] = pose.body.data[:, 0, idx, :2].data
        data[:, 2] = pose.body.confidence[:, 0, idx]
        return data

    def __call__(self, pose: Pose) -> np.ndarray:
        pose.normalize_distribution()
        data = []
        for joint in self.joints:
            component, point = COCO_TO_POSE_FORMAT[joint]
            data.append(self.__get_point(component, point, pose))
        # (num_landmarks, num_frames, 3) -> (num_frames, num_landmarks, 3)
        return np.array(data).transpose((1, 0, 2))


class SLGCNPad:
    """Pad/cắt về đúng num_frames rồi đổi trục -> (C, T, V, M) cho ONNX SL-GCN."""

    def __init__(self, num_frames: int = SLGCN_NUM_FRAMES) -> None:
        self.num_frames = num_frames

    def __call__(self, data: np.ndarray) -> np.ndarray:
        padded_data = np.zeros(
            (self.num_frames, data.shape[1], data.shape[2], 1),
            dtype=np.float32,
        )
        L = data.shape[0]
        if L < self.num_frames:
            padded_data[:L, :, :, 0] = data
            rest = self.num_frames - L
            num = int(np.ceil(rest / L))
            pad = np.concatenate([data for _ in range(num)], 0)[:rest]
            padded_data[L:, :, :, 0] = pad
        else:
            padded_data[:, :, :, 0] = data[:self.num_frames, :, :]
        # (num_frames, num_points, num_channels, num_people)
        # -> (num_channels, num_frames, num_points, num_people)
        return np.transpose(padded_data, [2, 0, 1, 3])


def read_labels(path: Path) -> dict[int, str]:
    id2label: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                id2label[int(row[0])] = row[1]
    return id2label


print("Đang nạp 2 model nhận dạng (SPOTER + SL-GCN) + nhãn...")
ID2LABEL = read_labels(GLOSS_CSV)
# Giúp onnxruntime-gpu tìm thấy CUDA/cuDNN cài qua pip (các gói nvidia-*-cu12 trong venv).
if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls()
    except Exception:  # noqa: BLE001
        pass
# Ưu tiên GPU (CUDA) cho ONNX, tự động lùi về CPU nếu không có.
_available = ort.get_available_providers()
_providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in _available]
SPOTER_SESSION = ort.InferenceSession(str(SPOTER_ONNX_PATH), providers=_providers)
SLGCN_SESSION = ort.InferenceSession(str(SLGCN_ONNX_PATH), providers=_providers)
print(f"ONNX providers — SPOTER: {SPOTER_SESSION.get_providers()} | SL-GCN: {SLGCN_SESSION.get_providers()}")

# Trích Pose (MediaPipe Holistic) MỘT lần / segment, dùng chung cho cả hai nhánh.
POSE_EXTRACT = PoseExtract()
# Nhánh SPOTER: nhận Pose -> tensor (70, 54, 2). PoseExtract đã tách riêng ở trên.
SPOTER_TRANSFORMS = Compose(
    [
        JointSelect(),
        TensorToDict(),
        SingleBodyDictNormalize(),
        SPOTERSingleHandDictNormalize(),
        DictToTensor(),
        Shift(),
        Pad(SPOTER_NUM_FRAMES),
    ]
)
# Nhánh SL-GCN: nhận Pose -> mảng (3, 150, 27, 1).
SLGCN_TRANSFORMS = Compose(
    [
        SLGCNJointSelect(SLGCN_NUM_POINTS),
        SLGCNPad(SLGCN_NUM_FRAMES),
    ]
)
print(f"Sẵn sàng — {len(ID2LABEL)} lớp gloss | fusion W_SLGCN={W_SLGCN:.2f}.")

# LLM / TTS nạp lười (tốn tài nguyên) — chỉ khởi tạo khi dùng lần đầu.
_llm_client = None
_tts_engine = None


def _get_llm():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI

        _llm_client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    return _llm_client


def _get_tts():
    global _tts_engine
    if _tts_engine is None:
        from vieneu import Vieneu

        print("Đang nạp VieNeu-TTS (lần đầu có thể lâu)...")
        _tts_engine = Vieneu(mode="standard")
    return _tts_engine


# --- Nhận dạng ---

def read_frames(path: str):
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
        raise gr.Error("Không đọc được khung hình nào từ video.")
    h, w = frames[0].shape[:2]
    return frames, float(fps), int(w), int(h)


def classify_frames(frames, fps, w, h):
    """Danh sách frame RGB -> (dict top-5 {gloss: prob}, gloss top-1).

    LATE-FUSION SPOTER + SL-GCN: trích Pose 1 lần rồi rẽ 2 nhánh tiền xử lý riêng,
    chạy 2 session ONNX, hợp nhất Ở TẦNG XÁC SUẤT:
        P_fused = W_SLGCN * P_slgcn + (1 - W_SLGCN) * P_spoter
    Cả hai ONNX đã xuất sẵn "probabilities" (softmax trong graph) -> KHÔNG softmax lại.
    Dùng chung cho cả mode video và mode realtime.
    """
    # PoseExtract dùng chung (MediaPipe Holistic) — chạy MỘT lần cho cả 2 nhánh.
    pose = POSE_EXTRACT({"frames": frames, "fps": fps, "width": w, "height": h})

    # Nhánh SPOTER trước (chỉ đọc pixel thô, không mutate Pose).
    spoter_tensor = SPOTER_TRANSFORMS(pose)
    spoter_in = spoter_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)  # (1, 70, 54, 2)
    p_spoter = SPOTER_SESSION.run(["probabilities"], {"poses": spoter_in})[0][0]

    # Nhánh SL-GCN (SLGCNJointSelect gọi normalize_distribution -> mutate Pose, nên chạy SAU).
    slgcn_arr = SLGCN_TRANSFORMS(pose)                                       # (3, 150, 27, 1)
    slgcn_in = slgcn_arr[None].astype(np.float32)                           # (1, 3, 150, 27, 1)
    p_slgcn = SLGCN_SESSION.run(["probabilities"], {"poses": slgcn_in})[0][0]

    # Fusion: chỉ còn phép cộng có trọng số (không softmax lại).
    probs = W_SLGCN * p_slgcn + (1.0 - W_SLGCN) * p_spoter
    top5 = np.argsort(probs)[::-1][:5]
    scores = {ID2LABEL[int(i)]: float(probs[i]) for i in top5}
    top_gloss = ID2LABEL[int(top5[0])]
    return scores, top_gloss


def recognize(video_path, signs, multi):
    """Video -> gloss. Nếu `multi`, tự tách video thành nhiều ký hiệu (theo tay lên/xuống)
    và nối tất cả; nếu không, coi cả video là MỘT ký hiệu (như trước)."""
    if not video_path:
        raise gr.Error("Hãy quay hoặc tải lên một video ký hiệu.")
    frames, fps, w, h = read_frames(video_path)

    if multi:
        segments = segment_video(frames)
        if not segments:
            raise gr.Error(
                "Không tách được ký hiệu nào. Hãy hạ tay xuống giữa các từ, hoặc bỏ chọn "
                "'Video có nhiều từ' nếu video chỉ có một ký hiệu."
            )
        new_signs = list(signs)
        scores = {}
        skipped = 0
        for seg_frames in segments:
            scores, top_gloss = classify_frames(seg_frames, fps, w, h)
            # Đoạn tự cắt có thể là rác (chuyển tiếp, tư thế nghỉ) -> chỉ nhận khi đủ tự tin.
            if scores.get(top_gloss, 0.0) < MIN_CONFIDENCE:
                skipped += 1
                continue
            new_signs.append(top_gloss)
        if skipped:
            gr.Warning(f"Đã bỏ qua {skipped} đoạn có độ tin cậy dưới {MIN_CONFIDENCE:.0%}.")
        return scores, new_signs, "  •  ".join(new_signs)

    scores, top_gloss = classify_frames(frames, fps, w, h)
    new_signs = signs + [top_gloss]
    return scores, new_signs, "  •  ".join(new_signs)


# --- LLM + TTS ---

_CONTROL_TOKEN_RE = re.compile(r"\s*/(?:no_)?think\b")
_LABEL_ONLY_LINE_RE = re.compile(
    r"^(?:bản dịch|câu(?:\s+dịch|\s+trả lời)?|kết quả|đáp án|trả lời|output|answer)"
    r"\s*[:：.\-–]*\s*$",
    flags=re.IGNORECASE,
)
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:bản dịch|câu(?:\s+dịch|\s+trả lời)?|kết quả|đáp án|trả lời|output|answer)"
    r"\s*[:：]\s*",
    flags=re.IGNORECASE,
)


def _strip_think(text: str) -> str:
    """Lam sach output LLM: bo khoi/tag suy nghi, token dieu khien va dong nhan thua.

    Qwen3 doi khi lam lo /no_think, /think va chen dong nhan (vd 'Bản dịch') vao output.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
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


def _dedupe_consecutive(signs: list[str]) -> list[str]:
    """Bỏ các từ trùng liền kề (cùng một ký hiệu bị nhận dạng nhiều lần)."""
    return [g for i, g in enumerate(signs) if i == 0 or g != signs[i - 1]]


def generate_sentence(signs: list[str]) -> str:
    system_prompt = (
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
        "Chỉ trả về câu kết quả, không giải thích, "
        "không dùng thẻ <think> hay định dạng đặc biệt."
    )
    client = _get_llm()
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Các từ nhận dạng được: {', '.join(signs)} /no_think"},
        ],
        temperature=0.3,
        max_tokens=128,  # cau ghep ngan; gioi han token de infer nhanh hon
    )
    return _strip_think(response.choices[0].message.content)


def make_sentence_and_speak(signs):
    """Danh sách từ -> câu tự nhiên (LLM) -> giọng nói (TTS)."""
    if not signs:
        raise gr.Error("Chưa có từ nào. Hãy nhận dạng ít nhất một ký hiệu trước.")

    # Cùng một ký hiệu bị nhận dạng lặp lại chỉ tính một lần khi ghép câu.
    signs = _dedupe_consecutive(signs)

    # 1) Sinh câu (nếu Ollama lỗi -> ghép từ thô).
    try:
        sentence = generate_sentence(signs)
    except Exception as exc:  # noqa: BLE001
        sentence = " ".join(signs)
        gr.Warning(f"LLM lỗi (kiểm tra Ollama đã chạy + pull {OLLAMA_MODEL}?): {exc}. Tạm ghép từ thô.")

    # Nếu LLM trả về rỗng (ví dụ toàn bộ nằm trong <think>), quay về ghép từ thô.
    if not sentence or not sentence.strip():
        sentence = " ".join(signs)

    # 2) Sinh giọng nói.
    try:
        engine = _get_tts()
        audio = np.asarray(engine.infer(sentence), dtype=np.float32).reshape(-1)
        if audio.size == 0:
            gr.Warning("TTS trả về audio rỗng — bỏ qua phần giọng nói.")
            return sentence, None
        return sentence, (engine.sample_rate, audio)
    except Exception as exc:  # noqa: BLE001
        gr.Warning(f"TTS lỗi: {exc}")
        return sentence, None


def clear_all():
    return [], "", "", None, None


# --- Realtime: tự động cắt ký hiệu theo cử động tay (dựa trên TBL) ---

_mp_pose = mp.solutions.pose


def _arm_angle(shoulder, elbow, wrist):
    """Góc khuỷu tay (vai-khuỷu-cổ tay), độ. Tay buông thẳng ~180, giơ/gập lại nhỏ hơn."""
    radians = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(
        shoulder[1] - elbow[1], shoulder[0] - elbow[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def _hands_up_from_pose(pose, frame_rgb, angle_threshold, visibility_threshold):
    """(tay đang giơ?, toạ độ cổ tay nhìn thấy được [chuẩn hoá 0..1]).

    "Đang giơ" = có ít nhất một tay góc khuỷu < ngưỡng + cổ tay đủ hiển thị.
    Danh sách cổ tay dùng để đo BIÊN ĐỘ CỬ ĐỘNG của cả đoạn — tay để yên (đặt lên
    bàn, buông hờ...) dù qua được ngưỡng góc vẫn không phải là đang ký hiệu.
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
    """Biên độ di chuyển lớn nhất (toạ độ chuẩn hoá) của chuỗi vị trí cổ tay."""
    pts = np.array([p for p in track if p is not None], dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.ptp(pts, axis=0).max())


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
):
    """Tách video NHIỀU ký hiệu thành danh sách các đoạn frame (mỗi đoạn ~1 ký hiệu).

    Chạy offline cùng logic tay lên/xuống với mode realtime (dùng MediaPipe Pose từng frame).
    Đoạn mà cổ tay gần như đứng yên (< min_wrist_motion, toạ độ chuẩn hoá) bị loại —
    đó là tư thế nghỉ lọt qua ngưỡng góc, không phải ký hiệu.
    """
    pose = _mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True
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
        # Đoạn còn dở ở cuối video (chưa kịp hạ tay).
        if active and len(buf) >= min_sign_frames and _track_motion(track) >= min_wrist_motion:
            segments.append(buf)
    finally:
        pose.close()
    return segments


class RealtimeSegmenter:
    """Máy trạng thái từng-frame: phát hiện lúc bắt đầu/kết thúc một ký hiệu.

    Tái dùng ý tưởng của Temporal Boundary Localization (góc cánh tay + độ hiển thị cổ tay):
    tay giơ lên (góc khuỷu < ngưỡng) = đang ký; tay hạ xuống = kết thúc. Có khử rung
    (min_up/min_down frame) và tiền-đệm (pre-roll) để không cắt cụt đầu ký hiệu.
    """

    def __init__(
        self,
        angle_threshold=150.0,
        visibility_threshold=0.6,
        min_up_frames=3,
        min_down_frames=8,
        min_sign_frames=8,
        max_sign_frames=90,
        preroll_frames=5,
        min_wrist_motion=0.04,
    ):
        self.angle_threshold = angle_threshold
        self.visibility_threshold = visibility_threshold
        self.min_up_frames = min_up_frames
        self.min_down_frames = min_down_frames
        self.min_sign_frames = min_sign_frames
        self.max_sign_frames = max_sign_frames
        self.min_wrist_motion = min_wrist_motion
        self.pose = _mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True
        )
        self.preroll = deque(maxlen=preroll_frames)
        self.reset_state()

    def reset_state(self):
        self.active = False           # đang trong một ký hiệu?
        self.up_count = 0             # số frame liên tiếp "tay lên"
        self.down_count = 0           # số frame liên tiếp "tay xuống"
        self.buffer = []              # frame của ký hiệu hiện tại
        self.wrist_track = []         # vị trí cổ tay từng frame (đo biên độ cử động)
        self.buf_t0 = None            # mốc thời gian frame đầu (để ước lượng fps)
        self.buf_t1 = None
        self.last_sign_time = None    # thời điểm ký hiệu gần nhất được nhận dạng
        self.spoken_upto = 0          # số từ đã tự động đọc (để không đọc lặp)

    def close(self):
        try:
            self.pose.close()
        except Exception:  # noqa: BLE001
            pass

    def _hands_up(self, frame_rgb):
        return _hands_up_from_pose(
            self.pose, frame_rgb, self.angle_threshold, self.visibility_threshold
        )

    def push(self, frame_rgb):
        """Nạp một frame. Trả về (frames, fps) khi vừa kết thúc một ký hiệu, ngược lại None."""
        now = time.time()
        up, wrists = self._hands_up(frame_rgb)
        wrist = tuple(np.mean(wrists, axis=0)) if wrists else None

        if not self.active:
            self.preroll.append(frame_rgb)
            if up:
                self.up_count += 1
                if self.up_count >= self.min_up_frames:
                    # Bắt đầu ký hiệu: mở đệm bằng pre-roll để giữ trọn phần đầu.
                    self.active = True
                    self.buffer = list(self.preroll)
                    self.wrist_track = [wrist]
                    self.buf_t0 = now
                    self.buf_t1 = now
                    self.down_count = 0
            else:
                self.up_count = 0
            return None

        # Đang trong một ký hiệu.
        self.buffer.append(frame_rgb)
        self.wrist_track.append(wrist)
        self.buf_t1 = now
        if up:
            self.down_count = 0
        else:
            self.down_count += 1

        finished = self.down_count >= self.min_down_frames or len(self.buffer) >= self.max_sign_frames
        if not finished:
            return None

        frames = self.buffer
        motion = _track_motion(self.wrist_track)
        dur = max(self.buf_t1 - self.buf_t0, 1e-3)
        fps = float(len(frames)) / dur if len(frames) > 1 else 25.0
        fps = float(np.clip(fps, 5.0, 60.0))
        # Reset về trạng thái nghỉ, sẵn sàng cho ký hiệu tiếp theo.
        self.active = False
        self.up_count = 0
        self.down_count = 0
        self.buffer = []
        self.wrist_track = []
        self.preroll.clear()
        if len(frames) < self.min_sign_frames:
            return None  # đoạn quá ngắn -> nhiều khả năng là nhiễu
        if motion < self.min_wrist_motion:
            return None  # tay gần như đứng yên -> tư thế nghỉ, không phải ký hiệu
        return frames, fps


def realtime_reset():
    """Xoá trạng thái mode realtime (kể cả bộ phân đoạn)."""
    return [], "🟢 Sẵn sàng — hãy bắt đầu ký hiệu.", "", "", None, None


def realtime_step(frame, signs, segmenter, angle_thr, min_down, idle_secs):
    """Handler streaming: nạp 1 frame webcam, tự cắt & phân loại khi ký hiệu kết thúc;
    khi nghỉ tay đủ lâu (idle_secs) và có từ mới -> tự ghép câu (LLM) + đọc (TTS)."""
    # Khởi tạo lười segmenter theo từng phiên (giữ trong gr.State).
    if segmenter is None:
        segmenter = RealtimeSegmenter()
    segmenter.angle_threshold = float(angle_thr)
    segmenter.min_down_frames = int(min_down)

    now = time.time()
    new_sign = False
    low_conf = None

    if frame is not None:
        result = segmenter.push(frame)
        if result is not None:
            frames, fps = result
            h, w = frame.shape[:2]
            try:
                scores, top_gloss = classify_frames(frames, fps, int(w), int(h))
                conf = scores.get(top_gloss, 0.0)
                if conf >= MIN_CONFIDENCE:
                    signs = signs + [top_gloss]
                    segmenter.last_sign_time = now
                    new_sign = True
                else:
                    # Đoạn rác (nghỉ tay, chuyển tiếp) -> bỏ qua thay vì thêm từ sai.
                    low_conf = (top_gloss, conf)
            except Exception as exc:  # noqa: BLE001
                gr.Warning(f"Lỗi phân loại đoạn vừa cắt: {exc}")

    signs_text = "  •  ".join(signs)

    # Trạng thái hiển thị.
    if segmenter.active:
        status = "✍️ Đang ký..."
    elif new_sign:
        status = f"✅ Nhận dạng: **{signs[-1]}**"
    elif low_conf:
        status = f"🤔 Bỏ qua đoạn không rõ (đoán *{low_conf[0]}* chỉ {low_conf[1]:.0%})."
    elif signs:
        status = "🟢 Nghỉ — chờ ký hiệu tiếp theo."
    else:
        status = "🟢 Sẵn sàng — hãy bắt đầu ký hiệu."

    # Tự động ghép câu + đọc khi nghỉ tay đủ lâu và còn từ chưa đọc.
    pending = len(signs) > segmenter.spoken_upto
    idle_ok = (
        not segmenter.active
        and segmenter.last_sign_time is not None
        and (now - segmenter.last_sign_time) >= float(idle_secs)
    )
    if pending and idle_ok:
        segmenter.spoken_upto = len(signs)  # đánh dấu trước để không kích hoạt lại
        sentence, audio = make_sentence_and_speak(signs)
        status = f"🗣️ Tự đọc câu: {sentence}"
        return signs, status, signs_text, segmenter, sentence, audio

    # Không có gì mới để đọc -> giữ nguyên ô câu/âm thanh (không phát lại).
    return signs, status, signs_text, segmenter, gr.skip(), gr.skip()


def perword_step(frame, signs, segmenter, angle_thr, min_down):
    """Handler cho tab 'Từng từ': như realtime nhưng KHÔNG tự đọc (idle vô hiệu hoá).

    Người dùng ký một từ -> hạ tay -> chốt từ đó -> lặp lại; ghép câu bằng nút bấm.
    """
    signs, status, signs_text, segmenter, _s, _a = realtime_step(
        frame, signs, segmenter, angle_thr, min_down, idle_secs=float("inf")
    )
    return signs, status, signs_text, segmenter


# --- Giao diện ---
_examples = [
    str(ROOT / "test" / name)
    for name in ("cam-on.mp4", "anh.mp4", "em.mp4")
    if (ROOT / "test" / name).exists()
]

with gr.Blocks(title="SignSpeak — Demo dịch ngôn ngữ ký hiệu", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🤟 SignSpeak — Demo dịch ngôn ngữ ký hiệu tiếng Việt\n"
        "Ba chế độ: **Video** (quay/tải lên; hỗ trợ 1 hoặc nhiều từ), "
        "**Realtime** (webcam liên tục, tự cắt + tự đọc khi nghỉ tay), và "
        "**Từng từ** (webcam: ký một từ rồi hạ tay để chốt, lặp lại).\n"
        "*Mẹo: ngồi lùi thấy nửa người trên + hai tay, nền gọn, đủ sáng.*"
    )

    with gr.Tabs():
        # ===== TAB 1: Video từng ký hiệu (cách cũ) =====
        with gr.Tab("🎬 Video từng ký hiệu"):
            signs_state = gr.State([])
            with gr.Row():
                with gr.Column():
                    video_in = gr.Video(
                        sources=["upload", "webcam"],
                        label="Video ký hiệu (~2-3 giây)",
                    )
                    multi_checkbox = gr.Checkbox(
                        value=False,
                        label="Video có nhiều từ (tự động tách theo cử động tay — nhớ hạ tay giữa các từ)",
                    )
                    recognize_btn = gr.Button("🔍 Nhận dạng ký hiệu", variant="primary")
                    gloss_out = gr.Label(num_top_classes=5, label="Kết quả nhận dạng (top-5)")
                    if _examples:
                        gr.Examples(examples=_examples, inputs=video_in, label="Ví dụ mẫu")

                with gr.Column():
                    signs_box = gr.Textbox(
                        label="Các từ đã nhận dạng (theo thứ tự)", interactive=False, lines=2
                    )
                    with gr.Row():
                        speak_btn = gr.Button("🔊 Tạo câu & đọc", variant="primary")
                        clear_btn = gr.Button("🗑️ Xóa")
                    sentence_out = gr.Textbox(label="Câu tiếng Việt (Qwen3)", lines=2)
                    audio_out = gr.Audio(label="Giọng nói (VieNeu-TTS)", autoplay=True)

            recognize_btn.click(
                recognize,
                inputs=[video_in, signs_state, multi_checkbox],
                outputs=[gloss_out, signs_state, signs_box],
            )
            speak_btn.click(
                make_sentence_and_speak,
                inputs=[signs_state],
                outputs=[sentence_out, audio_out],
            )
            clear_btn.click(
                clear_all,
                inputs=None,
                outputs=[signs_state, signs_box, sentence_out, audio_out, gloss_out],
            )

        # ===== TAB 2: Realtime (webcam liên tục, tự động cắt ký hiệu) =====
        with gr.Tab("⚡ Realtime"):
            gr.Markdown(
                "Bật webcam rồi **ký hiệu liên tục**: giơ tay lên để bắt đầu, hạ tay xuống "
                "để kết thúc — hệ thống tự cắt từng ký hiệu và nhận dạng ngay. Ký xong hết "
                "thì bấm **Tạo câu & đọc**."
            )
            rt_signs_state = gr.State([])
            rt_seg_state = gr.State(None)
            with gr.Row():
                with gr.Column():
                    rt_cam = gr.Image(
                        sources=["webcam"], streaming=True, type="numpy", label="Webcam"
                    )
                    rt_status = gr.Markdown("🟢 Sẵn sàng — hãy bắt đầu ký hiệu.")
                    with gr.Accordion("⚙️ Tinh chỉnh phát hiện ký hiệu", open=False):
                        rt_angle = gr.Slider(
                            90, 175, value=150, step=5,
                            label="Ngưỡng góc khuỷu (nhỏ hơn = tay phải giơ cao/gập hơn mới tính là 'đang ký')",
                        )
                        rt_min_down = gr.Slider(
                            2, 20, value=8, step=1,
                            label="Số frame tay hạ liên tục để chốt kết thúc ký hiệu",
                        )
                        rt_idle = gr.Slider(
                            1.0, 8.0, value=3.0, step=0.5,
                            label="Giây nghỉ tay để tự động ghép câu + đọc",
                        )
                with gr.Column():
                    rt_signs_box = gr.Textbox(
                        label="Các từ đã nhận dạng (theo thứ tự)", interactive=False, lines=2
                    )
                    with gr.Row():
                        rt_speak_btn = gr.Button("🔊 Tạo câu & đọc", variant="primary")
                        rt_clear_btn = gr.Button("🗑️ Xóa")
                    rt_sentence_out = gr.Textbox(label="Câu tiếng Việt (Qwen3)", lines=2)
                    rt_audio_out = gr.Audio(label="Giọng nói (VieNeu-TTS)", autoplay=True)

            rt_cam.stream(
                realtime_step,
                inputs=[rt_cam, rt_signs_state, rt_seg_state, rt_angle, rt_min_down, rt_idle],
                outputs=[rt_signs_state, rt_status, rt_signs_box, rt_seg_state, rt_sentence_out, rt_audio_out],
                stream_every=0.1,
                concurrency_limit=1,
                show_progress="hidden",
            )
            rt_speak_btn.click(
                make_sentence_and_speak,
                inputs=[rt_signs_state],
                outputs=[rt_sentence_out, rt_audio_out],
            )
            rt_clear_btn.click(
                realtime_reset,
                inputs=None,
                outputs=[rt_signs_state, rt_status, rt_signs_box, rt_sentence_out, rt_audio_out, rt_seg_state],
            )

        # ===== TAB 3: Từng từ (webcam, chủ động hạ tay sau mỗi từ) =====
        with gr.Tab("✋ Từng từ (hạ tay)"):
            gr.Markdown(
                "Bật webcam. Với **mỗi từ**: giơ tay lên ký → **hạ tay xuống** để chốt từ đó "
                "→ ký từ tiếp theo. Mỗi lần hạ tay là một từ được nhận dạng. Ký xong tất cả "
                "thì bấm **Tạo câu & đọc** (chế độ này KHÔNG tự đọc)."
            )
            pw_signs_state = gr.State([])
            pw_seg_state = gr.State(None)
            with gr.Row():
                with gr.Column():
                    pw_cam = gr.Image(
                        sources=["webcam"], streaming=True, type="numpy", label="Webcam"
                    )
                    pw_status = gr.Markdown("🟢 Sẵn sàng — ký một từ rồi hạ tay.")
                    with gr.Accordion("⚙️ Tinh chỉnh phát hiện ký hiệu", open=False):
                        pw_angle = gr.Slider(
                            90, 175, value=150, step=5,
                            label="Ngưỡng góc khuỷu (nhỏ hơn = tay phải giơ cao/gập hơn mới tính là 'đang ký')",
                        )
                        pw_min_down = gr.Slider(
                            2, 20, value=6, step=1,
                            label="Số frame tay hạ liên tục để chốt một từ",
                        )
                with gr.Column():
                    pw_signs_box = gr.Textbox(
                        label="Các từ đã nhận dạng (theo thứ tự)", interactive=False, lines=2
                    )
                    with gr.Row():
                        pw_speak_btn = gr.Button("🔊 Tạo câu & đọc", variant="primary")
                        pw_clear_btn = gr.Button("🗑️ Xóa")
                    pw_sentence_out = gr.Textbox(label="Câu tiếng Việt (Qwen3)", lines=2)
                    pw_audio_out = gr.Audio(label="Giọng nói (VieNeu-TTS)", autoplay=True)

            pw_cam.stream(
                perword_step,
                inputs=[pw_cam, pw_signs_state, pw_seg_state, pw_angle, pw_min_down],
                outputs=[pw_signs_state, pw_status, pw_signs_box, pw_seg_state],
                stream_every=0.1,
                concurrency_limit=1,
                show_progress="hidden",
            )
            pw_speak_btn.click(
                make_sentence_and_speak,
                inputs=[pw_signs_state],
                outputs=[pw_sentence_out, pw_audio_out],
            )
            pw_clear_btn.click(
                realtime_reset,
                inputs=None,
                outputs=[pw_signs_state, pw_status, pw_signs_box, pw_sentence_out, pw_audio_out, pw_seg_state],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
