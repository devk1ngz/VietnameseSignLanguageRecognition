# SignSpeak Frontend

Giao diện web (React + Vite) cho hệ thống dịch ngôn ngữ ký hiệu tiếng Việt (VSL) SignSpeak.
Dùng MediaPipe Holistic trong trình duyệt để trích keypoint từ webcam, gửi qua WebSocket tới
backend FastAPI để nhận dạng ký hiệu, ghép câu và phát giọng nói.

```
webcam → MediaPipe Holistic (in-browser) → 108 float keypoints → ws://.../ws/keypoints
       ← gloss / câu tiếng Việt / audio (WAV bytes)
```

## Cấu trúc

```text
src/
├── App.jsx                 # layout, khởi tạo WebSocket
├── main.jsx                # entry point
├── components/
│   ├── CameraPanel/        # camera + gửi frame
│   ├── ResultPanel/        # hiển thị gloss / câu / audio
│   ├── ManualPanel/        # chế độ nhận dạng từng từ
│   ├── LandingPage/
│   └── ui/                 # component dùng chung
├── hooks/                   # useCamera, useWebSocket
├── services/                # ws.js, api.js — nơi duy nhất gọi network
├── store/                   # Zustand store
└── utils/                   # frameCapture, audioHelper
```

Quy tắc cho AI coding agent: xem `AGENT.md`.

## Cài đặt & chạy

```bash
npm install
npm run dev       # http://localhost:5173
npm run build      # build production
npm run lint       # oxlint
```

## Cấu hình môi trường

Sao chép `.env.development` hoặc `.env.production` phù hợp, hoặc chỉnh trực tiếp:

```bash
VITE_WS_URL=ws://localhost:8000/ws/keypoints
VITE_API_URL=http://localhost:8000
```

Backend serving cần chạy trước (xem `backend/README.md`).
