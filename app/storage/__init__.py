"""Where the service keeps files: one interface, one place that picks the backend.

Business code calls get_storage() and nothing else; it never checks which
backend is active.
"""

from functools import lru_cache
from typing import Optional

from app import config
from app.storage.base import Storage, StorageNotFound
from app.storage.local import LocalStorage

__all__ = ["Storage", "StorageNotFound", "LocalStorage", "get_storage"]


def get_storage() -> Storage:
    """Build the backend selected by STORAGE_BACKEND.

    The settings are read on every call, like the rest of app.config, so
    tests can switch backends with monkeypatch. Building a LocalStorage is
    just a Path; Blob clients are cached per configuration.
    """
    backend = config.storage_backend()
    if backend == "local":
        return LocalStorage(config.jobs_dir())
    if backend == "blob":
        connection_string = config.azure_storage_connection_string()
        if not connection_string:
            raise ValueError(
                "STORAGE_BACKEND=blob needs AZURE_STORAGE_CONNECTION_STRING"
            )
        return _blob_storage(
            connection_string,
            config.storage_container(),
            config.azure_storage_public_endpoint() or None,
        )
    raise ValueError(
        "unknown STORAGE_BACKEND {!r} (expected: local, blob)".format(backend)
    )


@lru_cache(maxsize=4)
def _blob_storage(connection_string: str, container: str,
                  public_endpoint: Optional[str]) -> Storage:
    """One client per distinct configuration. Building a BlobStorage makes a
    network call (it ensures the container exists), which should not happen
    on every request; keying the cache on the settings keeps env changes in
    tests working."""
    # Imported here so the Azure SDK is only needed when Blob is used.
    from app.storage.blob import BlobStorage

    return BlobStorage(connection_string, container, public_endpoint=public_endpoint)
