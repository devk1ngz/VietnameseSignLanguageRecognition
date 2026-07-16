#!/usr/bin/env bash
#
# run_demo.sh — Khởi chạy web demo SignSpeak chỉ bằng một lệnh.
#
#   ./run_demo.sh
#
# Việc script làm:
#   1) (Tuỳ chọn) Đảm bảo Ollama chạy cho bước ghép câu (LLM). Ưu tiên Ollama cài
#      NATIVE (`ollama serve`); nếu không có mới dùng Docker. Tự tải model lần đầu
#      nếu chưa có. KHÔNG có Ollama vẫn chạy được demo — chỉ là câu sẽ là các từ
#      ghép thô thay vì câu tự nhiên.
#   2) Chạy Gradio demo và mở trình duyệt tại http://localhost:7860.
#
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/backend"   # AI core + serving + demo (truoc day la recognition/)
PORT="${PORT:-7860}"
URL="http://localhost:${PORT}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:1.7b}"

# uv phải chạy đúng venv của backend/ — bỏ VIRTUAL_ENV nếu shell có sẵn (xem CLAUDE.md).
unset VIRTUAL_ENV 2>/dev/null || true

echo "🤟 SignSpeak — khởi chạy web demo"

have() { command -v "$1" >/dev/null 2>&1; }

ollama_ready() {
  have curl && curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1
}

# Chờ Ollama sẵn sàng (tối đa ~30s).
wait_ollama() {
  have curl || return 0
  for _ in $(seq 1 30); do ollama_ready && return 0; sleep 1; done
  return 1
}

# --- Bước 1 (tuỳ chọn): đảm bảo Ollama sẵn sàng cho bước ghép câu. ---
# Ưu tiên: (a) đã chạy sẵn -> (b) Ollama native -> (c) Docker -> (d) bỏ qua.
OLLAMA_PULL=""   # cách chạy `ollama pull` tương ứng với backend đang dùng
if ollama_ready; then
  echo "→ Ollama đã chạy (${OLLAMA_URL})."
  have ollama && OLLAMA_PULL="ollama"
elif have ollama; then
  echo "→ Ollama native chưa chạy, đang bật 'ollama serve' ở nền..."
  ollama serve >/tmp/ollama-serve.log 2>&1 &
  wait_ollama && echo "→ Ollama đã sẵn sàng." \
    || echo "⚠️  Ollama chưa lên — demo vẫn chạy, câu sẽ là ghép từ thô."
  OLLAMA_PULL="ollama"
elif have docker; then
  echo "→ Không có Ollama native, dùng Docker: đang bật container..."
  docker compose -f "$ROOT_DIR/docker-compose.yml" up -d ollama \
    || echo "⚠️  Không bật được Ollama — demo vẫn chạy, câu sẽ là ghép từ thô."
  wait_ollama && echo "→ Ollama đã sẵn sàng." || true
  OLLAMA_PULL="docker compose -f $ROOT_DIR/docker-compose.yml exec -T ollama ollama"
else
  echo "→ Không có Ollama lẫn Docker — bỏ qua LLM (câu = ghép từ thô). Nhận dạng + TTS vẫn chạy."
fi

# Tải model nếu chưa có (chỉ lần đầu, có thể lâu).
if [ -n "$OLLAMA_PULL" ] && ollama_ready; then
  if have curl && curl -sf "${OLLAMA_URL}/api/tags" 2>/dev/null | grep -q "\"${OLLAMA_MODEL}\""; then
    echo "→ Model ${OLLAMA_MODEL} đã sẵn sàng."
  else
    echo "→ Chưa có model ${OLLAMA_MODEL}, đang tải (lần đầu, có thể lâu)..."
    $OLLAMA_PULL pull "${OLLAMA_MODEL}" \
      || echo "⚠️  Không tải được model — câu sẽ là ghép từ thô."
  fi
fi

# --- Bước 2: chạy Gradio demo + mở trình duyệt. ---
echo "→ Mở web demo tại ${URL}  (nhấn Ctrl+C để dừng)"
if have xdg-open; then
  ( sleep 5; xdg-open "${URL}" >/dev/null 2>&1 || true ) &
fi

cd "$APP_DIR"
exec uv run python demo_gradio.py
