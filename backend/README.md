# SignSpeak backend — VSL-400 serving

Phiên bản **serving** của hệ thống dịch ngôn ngữ ký hiệu tiếng Việt (VSL):
FastAPI + demo Gradio chạy 2 model ONNX (late-fusion SPOTER + SL-GCN), ghép câu bằng
Qwen3 (Ollama) và đọc bằng VieNeu-TTS.

> Code training / evaluation / export ONNX nằm ở **repo training riêng** — repo này chỉ
> tiêu thụ artifact đã export.

## Cấu trúc

```text
pyproject.toml               # uv-managed (Python 3.12, torch cu126)
demo_gradio.py               # demo Gradio trọn gói (không cần frontend)
serving/                     # FastAPI: WS /ws/keypoints + REST
src/
  ├── pipelines/spoter_graph_classification.py   # transform tiền xử lý SPOTER
  └── utils/constants.py                         # SLGCN_JOINTS, COCO_TO_POSE_FORMAT
models/
  ├── spoter/spoter_v3.onnx                      # (git-untracked)
  └── sl-gcn/sl_gcn_ensemble.onnx                # (git-untracked)
experiments/gloss.csv        # 400 nhãn gloss (id,gloss — git-untracked)
```

## Cài đặt & chạy

```bash
uv sync                                                     # cần GPU NVIDIA (torch cu126)
./scripts/download_weights.sh                               # kiểm tra artifact tồn tại
uv run uvicorn serving.main:app --host 0.0.0.0 --port 8000  # serving cho frontend React
uv run python demo_gradio.py                                # hoặc demo Gradio ở :7860
```

Bước ghép câu cần Ollama tại `OLLAMA_BASE_URL` (mặc định `http://localhost:11434`) với model
`OLLAMA_MODEL` (mặc định `qwen3:1.7b`). Thiếu Ollama thì nhận dạng + TTS vẫn chạy, câu sẽ là
gloss thô ghép lại.

Chi tiết kiến trúc: `ARCHITECTURE.md`, `docs/local_fusion_deployment.md`.
