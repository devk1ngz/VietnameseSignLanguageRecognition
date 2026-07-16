# SignSpeak Backend

Backend FastAPI: keypoint stream -> gloss (SPOTER/ONNX) -> cau tieng Viet (Qwen3/Ollama)
-> giong noi (VieNeu-TTS) -> WAV bytes.

Xem `ARCHITECTURE.md` (thiet ke) va `AGENT.md` (quy tac cho AI agent).

## Kien truc thuc te

Backend la **lop phuc vu mong** tren ma trong `recognition/`:

- Nhan dang chay bang **ONNX** + nhan `experiments/gloss.csv`, khong phai checkpoint `.pth`.
  Hai model (deu xuat san `probabilities`): `models/spoter/spoter_v3.onnx` (SPOTER) va
  `models/sl-gcn/sl_gcn_ensemble.onnx` (SL-GCN).
- **Luong video-file** (`POST /api/recognize-video`): server tu chay MediaPipe Holistic,
  co Pose day du nen dung **late-fusion SPOTER + SL-GCN** giong `demo_gradio.py`
  (`P = 0.75*P_slgcn + 0.25*P_spoter`, KHONG softmax lai).
- **Luong realtime/REST keypoint** (`WS /ws/keypoints`, `POST /api/recognize`): client gui
  keypoint **da qua MediaPipe Holistic + chon khop**: `keypoints` (108 = 54 khop SPOTER x,y) VA
  `slgcn_keypoints` (81 = 27 khop SL-GCN x,y,confidence). Backend chay **late-fusion** giong luong
  video. SPOTER: chuoi chuan hoa `TensorToDict -> ... -> Pad` (tai lai tu
  `src/pipelines/spoter_graph_classification.py` qua `serving/recognition_bridge.py`). SL-GCN:
  `serving/services/slgcn.preprocess_snapshot()` tai hien normalize_distribution + pad tu keypoint
  tho (da kiem chung **bit-exact** voi AI core). Neu client cu KHONG gui `slgcn_keypoints` ->
  tu dong lui ve **SPOTER-only** (nguong `spoter_confidence_threshold`); khi fuse dung
  `fusion_confidence_threshold` (thap hon vi phan phoi fusion phang hon).
- LLM goi Ollama `/api/chat` (`httpx`), prompt tieng Viet + `/no_think` lay tu demo.
- TTS dung `vieneu`, tra WAV bytes in-memory (khong ghi file).

### Hop dong keypoint (108 float / frame)

54 khop x (x, y), **toa do pixel**, theo thu tu `JointSelect`: 12 khop than
(nose, neck, rightEye, leftEye, rightEar, leftEar, rightShoulder, leftShoulder,
rightElbow, leftElbow, rightWrist, leftWrist) roi 21 khop tay TRAI roi 21 khop tay PHAI.
**Khop index 1 (neck) phai la `[0, 0]`.**

## Chay local

Dung chung moi truong voi `recognition/` (da co gan du dependency nang):

```bash
# 1. Dong bo moi truong recognition (da khai bao san pydantic-settings, websockets, pytest)
cd ../recognition && uv sync && cd ../backend

# 2. Tao .env (tuy chon; mac dinh da chay duoc)
cp .env.example .env

# 3. Dam bao Ollama chay + da pull qwen3:1.7b (cho buoc ghep cau)
ollama serve &
ollama pull qwen3:1.7b

# 4. Chay server
../recognition/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Khong co Ollama/TTS thi nhan dang van chay; cau se ghep tu tho va bo qua giong noi.

## Kiem thu

```bash
./scripts/download_weights.sh   # kiem tra artifact (ONNX + gloss.csv) ton tai
```

## API

- `WS /ws/keypoints` -- nhan `{"keypoints": [108 float], "end_sign": bool}`; tra JSON
  `gloss`/`processing`/`sentence`/`error` va **binary WAV** sau `sentence`.
- `POST /api/recognize` -- fallback one-shot: `{"frames": [[108 float], ...]}`.
- `POST /api/tts` -- `{"text": "..."}` -> `audio/wav` bytes.
- `GET /health` -- trang thai spoter/ollama/tts.

## Docker (tuy chon)

```bash
docker compose up --build   # build context la goc repo; can NVIDIA runtime
```
