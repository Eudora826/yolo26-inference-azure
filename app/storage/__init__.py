"""Where the service keeps files: one interface, one place that picks the backend.

Business code calls get_storage() and nothing else; it never checks which
backend is active.
"""

from app import config
from app.storage.base import Storage, StorageNotFound
from app.storage.local import LocalStorage

__all__ = ["Storage", "StorageNotFound", "LocalStorage", "get_storage"]


def get_storage() -> Storage:
    """Build the backend selected by STORAGE_BACKEND.

    Not cached: the settings are read on every call, like the rest of
    app.config, so tests can switch backends with monkeypatch. Building a
    LocalStorage is just a Path.
    """
    backend = config.storage_backend()
    if backend == "local":
        return LocalStorage(config.jobs_dir())
    raise ValueError(
        "unknown STORAGE_BACKEND {!r} (expected: local)".format(backend)
    )
