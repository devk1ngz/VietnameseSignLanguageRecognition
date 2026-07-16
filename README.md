# SignSpeak — Vietnamese Sign Language Translator

Hệ thống dịch **ngôn ngữ ký hiệu tiếng Việt** (VSL) thời gian thực: nhận dạng cử chỉ từ
video/webcam, ghép thành câu tiếng Việt tự nhiên và đọc thành giọng nói.

```
video / webcam → nhận dạng ký hiệu (AI) → ghép câu (LLM) → giọng nói (TTS)
```

## Tính năng

- 🎬 Nhận dạng từ **video** tải lên hoặc quay trực tiếp
- ⚡ Nhận dạng **realtime** qua webcam, tự tách từng ký hiệu
- 📝 Ghép các từ nhận dạng được thành **câu tiếng Việt tự nhiên**
- 🔊 Đọc câu bằng **giọng nói tiếng Việt**
- 🌐 Giao diện web (React) + API (FastAPI WebSocket/REST), kèm bản demo Gradio chạy độc lập

## Cấu trúc

| Thư mục | Vai trò |
|---|---|
| `backend/` | AI + API serving (Python, FastAPI, ONNX) và demo Gradio |
| `frontend/` | Giao diện web (React + Vite) |

## Chạy nhanh

```bash
# Demo trọn gói (không cần frontend) — mở http://localhost:7860
./run_demo.sh

# Hoặc chạy đầy đủ backend + frontend
cd backend && uv sync && uv run uvicorn serving.main:app --port 8000
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Yêu cầu: GPU NVIDIA (CUDA), Python 3.12 + [uv](https://docs.astral.sh/uv/), Node.js.
Bước ghép câu cần [Ollama](https://ollama.com) (tùy chọn — thiếu thì hệ thống vẫn chạy,
câu trả về là các từ ghép thô).

Chi tiết cài đặt và cấu hình: xem `backend/README.md` và `frontend/README.md`.
