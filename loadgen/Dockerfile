# The build context is the repo root (see build.context in docker-compose.yml),
# so the root-level cat.jpg can be copied in as the seed image.
FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless needs no libGL, but still links against libglib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY loadgen/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY loadgen/loadgen.py .
COPY cat.jpg .

RUN useradd -m appuser
USER appuser

# -u disables output buffering; otherwise docker compose logs shows nothing
# until the buffer fills up
CMD ["python", "-u", "loadgen.py"]
