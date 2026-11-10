"""Storage on the local filesystem: the default backend, and the behaviour the
service had before the storage interface existed."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

from app.storage.base import CHUNK_SIZE, StorageNotFound


class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        """Map a key to a path under root, rejecting anything that escapes it.

        Keys are built by the service today, but this is the one place that
        turns them into paths, so it is the one place to stop a key such as
        "../../etc/passwd" from ever reaching the filesystem.
        """
        if not key or key.startswith(("/", "\\")):
            raise ValueError("invalid storage key: {!r}".format(key))
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError("storage key escapes the root: {!r}".format(key))
        return path

    def put(self, key: str, data: BinaryIO) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename it into place, so a reader
        # never sees a half-written object and a crash leaves no partial file.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as out:
                shutil.copyfileobj(data, out, CHUNK_SIZE)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def get(self, key: str) -> Iterator[bytes]:
        try:
            # Opened here rather than inside the generator, so a missing key
            # fails now; _chunks closes the handle when iteration ends.
            handle = open(self._path(key), "rb")
        except FileNotFoundError:
            raise StorageNotFound(key) from None
        return _chunks(handle)

    def url(self, key: str, expires_in: int = 900) -> Optional[str]:
        # A file on this machine has no URL a client could use.
        return None

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


def _chunks(handle: BinaryIO) -> Iterator[bytes]:
    with handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                return
            yield chunk
