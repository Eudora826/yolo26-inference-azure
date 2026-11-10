"""The storage interface every backend implements.

Callers deal in keys ("<job_id>/input.mp4") and bytes, never in file paths or
SDK objects, so the backend can change (local disk, Azure Blob) without any
change to the code that uses it.
"""

from typing import BinaryIO, Iterator, Optional, Protocol

# 1 MiB: large enough that a 200 MB video is a few hundred reads, small
# enough that streaming one never holds much in memory.
CHUNK_SIZE = 1024 * 1024


class StorageNotFound(KeyError):
    """The key does not exist. Raised by every backend, so callers never see
    a backend-specific error such as FileNotFoundError."""


class Storage(Protocol):
    """Structural interface: any class with these methods is a Storage.

    A Protocol rather than an ABC, so a test double needs no base class; the
    shared contract tests in tests/test_storage.py are what keep every
    implementation honest.
    """

    def put(self, key: str, data: BinaryIO) -> None:
        """Store the contents of a binary stream under key, replacing any
        existing object."""
        ...

    def get(self, key: str) -> Iterator[bytes]:
        """Return the object's bytes in chunks.

        Raises StorageNotFound immediately (not on first iteration) when the
        key does not exist, so callers can turn it into a 404 before they
        start streaming a response.
        """
        ...

    def url(self, key: str, expires_in: int = 900) -> Optional[str]:
        """A time-limited URL a client can download from directly, or None if
        the backend has no such thing and the service must stream the bytes
        itself."""
        ...

    def delete(self, key: str) -> None:
        """Remove the object. Deleting a missing key is not an error."""
        ...
