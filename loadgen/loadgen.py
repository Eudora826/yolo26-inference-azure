"""Generates traffic for the demo stack.

Three reasons it exists, each more practical than the last:

1. Dashboards without traffic are empty, and empty screenshots in a README
   hurt more than they help.
2. Without traffic PromQL cannot be checked: rate() returns nothing, and you
   cannot tell whether the query is right.
3. Alert rules cannot be tested: to verify that "error rate above 5% fires",
   something has to actually send bad requests.

So this script deliberately produces **varied** traffic: several endpoints,
parameters and images, plus a share of **intentionally** bad requests.
Uniform traffic would leave half the dashboard panels flat.

Everything is configured through environment variables; see docker-compose.yml.
"""

from __future__ import annotations

import io
import logging
import os
import random
import signal
import sys
import tempfile
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("loadgen")

TARGET = os.getenv("TARGET_URL", "http://yolo-service:8000").rstrip("/")
RPS = float(os.getenv("RPS", "3"))
ERROR_RATE = float(os.getenv("ERROR_RATE", "0.06"))
VIDEO_EVERY = float(os.getenv("VIDEO_EVERY_SECONDS", "0"))
SEED_IMAGE = os.getenv("SEED_IMAGE", "cat.jpg")
API_KEY = os.getenv("API_KEY", "")

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# On SIGTERM, finish the current request and exit. Without this,
# docker compose down waits for its 10-second timeout and then SIGKILLs it.
_running = True


def _stop(signum, _frame):
    global _running
    _running = False
    log.info("received signal %s, exiting", signum)


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


# ---------------------------------------------------------------------------
# Startup race: wait until the service is actually available
#
# The compose healthcheck already guards this, but a robust client should not
# die on a single failed connection, and this way the script also runs
# outside compose.
# ---------------------------------------------------------------------------


def wait_for_service(timeout: float = 300.0) -> bool:
    deadline = time.monotonic() + timeout
    delay = 1.0
    while _running and time.monotonic() < deadline:
        try:
            if requests.get(f"{TARGET}/health", timeout=3).ok:
                log.info("service is ready: %s", TARGET)
                return True
        except requests.RequestException as exc:
            log.info("waiting for %s (%s)", TARGET, type(exc).__name__)
        time.sleep(delay)
        # Exponential backoff, so startup does not flood the log
        delay = min(delay * 1.6, 10.0)
    return False


# ---------------------------------------------------------------------------
# Images
#
# Sending the same image every time would pin detections_per_image to a
# single value and leave the class distribution with one bar. This builds
# variety from a single seed image.
# ---------------------------------------------------------------------------

_seed = Image.open(SEED_IMAGE).convert("RGB")


def _jpeg(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _variant_transformed() -> bytes:
    """Resize / flip / change brightness.

    Detections stay stable, while latency and confidence vary.
    """
    img = _seed.copy()
    scale = random.uniform(0.4, 1.3)
    img = img.resize((max(64, int(img.width * scale)), max(64, int(img.height * scale))))
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    return _jpeg(img)


def _variant_composite() -> bytes:
    """Paste the seed image 1-5 times onto a canvas to vary the detection count."""
    width, height = 960, 720
    background = tuple(random.randint(30, 220) for _ in range(3))
    canvas = Image.new("RGB", (width, height), background)
    for _ in range(random.randint(1, 5)):
        scale = random.uniform(0.15, 0.45)
        piece = _seed.resize((
            max(16, int(_seed.width * scale)),
            max(16, int(_seed.height * scale)),
        ))
        canvas.paste(
            piece,
            (random.randint(0, max(1, width - piece.width)),
             random.randint(0, max(1, height - piece.height))),
        )
    return _jpeg(canvas)


def _variant_noise() -> bytes:
    """Pure noise: zero detections, so the low detections_per_image buckets get data."""
    arr = np.random.randint(
        0, 255, (random.randint(240, 720), random.randint(320, 960), 3), dtype=np.uint8
    )
    return _jpeg(Image.fromarray(arr))


def random_image() -> bytes:
    roll = random.random()
    if roll < 0.65:
        return _variant_transformed()
    if roll < 0.88:
        return _variant_composite()
    return _variant_noise()


def make_video(path: str, frames: int = 60, size=(320, 240), fps: int = 15) -> None:
    """Synthesize a short video instead of committing a multi-MB binary to the repo."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    sprite = _seed.resize((size[0] // 2, size[1] // 2))
    sprite_bgr = cv2.cvtColor(np.array(sprite), cv2.COLOR_RGB2BGR)
    sprite_h, sprite_w = sprite_bgr.shape[:2]
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        x = int((size[0] - sprite_w) * (i / max(1, frames - 1)))
        frame[0:sprite_h, x:x + sprite_w] = sprite_bgr
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def _send(name: str, method: str, path: str, **kwargs) -> None:
    t0 = time.perf_counter()
    try:
        resp = requests.request(
            method, f"{TARGET}{path}", headers=HEADERS, timeout=30, **kwargs
        )
        log.info("%-18s %s %-22s %sms", name, resp.status_code, path,
                 round((time.perf_counter() - t0) * 1000))
    except requests.RequestException as exc:
        log.warning("%-18s ERR %-22s %s", name, path, type(exc).__name__)


def predict(annotated: bool = False) -> None:
    params = {"conf": round(random.uniform(0.15, 0.6), 2)}
    # Sometimes add a class filter, so /predict sees more than one parameter shape
    if random.random() < 0.25:
        params["classes"] = random.choice(["cat", "dog", "person", "cat,dog"])
    _send(
        "predict/annotated" if annotated else "predict",
        "POST",
        "/predict/annotated" if annotated else "/predict",
        params=params,
        files={"file": ("frame.jpg", random_image(), "image/jpeg")},
    )


def bad_request() -> None:
    """Intentionally bad requests: without them the error-rate panel stays at 0
    and the alerts cannot be tested."""
    kind = random.choice(["not-an-image", "unknown-class", "bad-conf"])
    if kind == "not-an-image":
        # 415
        _send("bad:not-image", "POST", "/predict",
              files={"file": ("notes.txt", b"definitely not an image", "text/plain")})
    elif kind == "unknown-class":
        # 400
        _send("bad:unknown-class", "POST", "/predict",
              params={"classes": "dragon"},
              files={"file": ("frame.jpg", random_image(), "image/jpeg")})
    else:
        # 422, FastAPI parameter validation
        _send("bad:conf-range", "POST", "/predict",
              params={"conf": 2.0},
              files={"file": ("frame.jpg", random_image(), "image/jpeg")})


def submit_video() -> None:
    path = os.path.join(tempfile.gettempdir(), "loadgen-clip.mp4")
    if not os.path.exists(path):
        make_video(path)
    with open(path, "rb") as fh:
        data = fh.read()
    _send("video:submit", "POST", "/videos",
          params={"conf": 0.3},
          files={"file": ("clip.mp4", data, "video/mp4")})


def one_request() -> None:
    if random.random() < ERROR_RATE:
        return bad_request()
    roll = random.random()
    if roll < 0.60:
        predict()
    elif roll < 0.85:
        predict(annotated=True)
    elif roll < 0.95:
        _send("health", "GET", "/health")
    else:
        _send("model/info", "GET", "/model/info")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    log.info("target %s | RPS=%s | error rate=%s | video every=%ss",
             TARGET, RPS, ERROR_RATE, VIDEO_EVERY or "off")

    if not wait_for_service():
        log.error("service was not ready before the timeout, exiting")
        return 1

    interval = 1.0 / RPS if RPS > 0 else 0.5
    # Advance by "when the next request is due" instead of a fixed
    # sleep(interval). A fixed sleep ignores the time the request itself
    # takes, so the real rate would always fall below the configured one.
    next_at = time.monotonic()
    next_video = time.monotonic() + VIDEO_EVERY if VIDEO_EVERY > 0 else float("inf")

    while _running:
        now = time.monotonic()
        if now < next_at:
            # Sleep in short slices so SIGTERM is handled promptly
            time.sleep(min(next_at - now, 0.25))
            continue

        next_at += interval
        if next_at < now:
            # Too far behind (e.g. the service stalled): reset instead of
            # bursting to catch up
            next_at = now + interval

        if now >= next_video:
            submit_video()
            next_video = now + VIDEO_EVERY
            continue

        one_request()

    log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
