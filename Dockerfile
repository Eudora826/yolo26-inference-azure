# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# The weights are fetched at build time and kept out of git: a 6.3 MB binary,
# once committed, stays in the history forever, and Ultralytics already
# hosts it on its Releases page.
#
# Downloading here (instead of copying a local file) means a fresh clone runs
# with `docker compose up --build` directly. An earlier version copied a file
# that .gitignore excluded, so the first README command failed for anyone who
# tried it, and CI never noticed because every job fetched the file first.
#
# ADD supports remote URLs natively, so curl or wget is not needed.
ARG WEIGHTS_URL=https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
ADD ${WEIGHTS_URL} /weights/yolov8n.pt

# ---- Stage 2: runtime ----
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -m appuser

COPY --chown=appuser:appuser app/ ./app/

COPY --from=builder --chown=appuser:appuser /weights/yolov8n.pt ./yolov8n.pt

# The commit this image was built from, reported in every /predict response.
# Declared last on purpose: a new value only rebuilds the layers after it,
# so the dependency layers above stay cached.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
