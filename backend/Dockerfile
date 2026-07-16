# Anh Docker cho ca AI core (src/) + lop serving (serving/) — mot moi truong duy nhat.
# Torch cu126, onnxruntime-gpu, mediapipe, pose-format, vieneu, fastapi/uvicorn...
# Build context = chinh thu muc nay (backend/):  docker compose up --build
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# uv de dong bo dependency (torch cu126 index da khai bao trong pyproject.toml).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Cai deps truoc (tan dung layer cache): chi can manifest + lockfile.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy toan bo source (src/, serving/, models/, experiments/... ; .venv host bi .dockerignore loai).
COPY . /app

EXPOSE 8000
# recognition_root mac dinh = /app (goc du an); khong can ENV rieng.
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "serving.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
