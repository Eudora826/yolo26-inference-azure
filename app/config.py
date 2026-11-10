"""Runtime configuration, read from environment variables.

Values are read at call time (not import time) so tests can override them
with monkeypatch and containers can configure the service without rebuilds.
"""

import os
import tempfile

SERVICE_VERSION = "2.0.0"


def model_path() -> str:
    """Path to the YOLO weights file."""
    return os.getenv("MODEL_PATH", "yolov8n.pt")


def max_upload_mb() -> int:
    """Maximum accepted upload size in megabytes."""
    return int(os.getenv("MAX_UPLOAD_MB", "20"))


def max_video_upload_mb() -> int:
    """Maximum accepted video upload size in megabytes."""
    return int(os.getenv("MAX_VIDEO_UPLOAD_MB", "200"))


def api_key() -> str:
    """If non-empty, inference endpoints require this key in X-API-Key."""
    return os.getenv("API_KEY", "")


def rate_limit_per_minute() -> int:
    """Per-client request limit for inference endpoints. 0 disables limiting."""
    return int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))


def storage_backend() -> str:
    """Where job files are kept: "local" (the default)."""
    return os.getenv("STORAGE_BACKEND", "local").strip().lower()


def jobs_dir() -> str:
    """Root directory of the local storage backend (video inputs and results)."""
    return os.getenv("JOBS_DIR", os.path.join(tempfile.gettempdir(), "yolo-jobs"))


def job_retention_minutes() -> int:
    """How long a finished job keeps its files. 0 keeps them forever."""
    return int(os.getenv("JOB_RETENTION_MINUTES", "60"))


def otlp_endpoint() -> str:
    """OTLP gRPC endpoint for traces. Empty disables tracing entirely.

    Deliberately the standard OpenTelemetry variable name, so anyone who
    has deployed an OTel app already knows how to configure this one.
    """
    return os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")


def service_name() -> str:
    """Resource name reported to the tracing backend.

    This is what the Jaeger service dropdown shows. Kept in sync with the
    value docker-compose.yml sets, so a local run and a containerised run
    land under the same name instead of creating two entries.
    """
    return os.getenv("OTEL_SERVICE_NAME", "yolo-inference-platform")


def service_version() -> str:
    return os.getenv("SERVICE_VERSION", SERVICE_VERSION)


def git_sha() -> str:
    """Commit the running code was built from.

    Baked into the image at build time (Dockerfile ARG GIT_SHA), because the
    image has neither a .git directory nor the git binary to ask at runtime.
    "unknown" means the build did not pass it, e.g. tests or a bare uvicorn.
    """
    return os.getenv("GIT_SHA", "") or "unknown"
