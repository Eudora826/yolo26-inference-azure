"""Contract tests for the storage backends.

Every backend must pass the same tests: that is what makes them
interchangeable. A new backend joins by adding one entry to the `storage`
fixture's params.
"""

import io

import pytest

from app.storage import LocalStorage, StorageNotFound, get_storage
from app.storage.base import CHUNK_SIZE


@pytest.fixture(params=["local"])
def storage(request, tmp_path):
    if request.param == "local":
        return LocalStorage(str(tmp_path / "store"))
    raise AssertionError("unknown backend " + request.param)


def read(storage, key: str) -> bytes:
    return b"".join(storage.get(key))


# ---------------------------------------------------------------------------
# Contract: every backend
# ---------------------------------------------------------------------------


def test_put_then_get_returns_the_same_bytes(storage):
    storage.put("job1/input.mp4", io.BytesIO(b"hello video"))
    assert read(storage, "job1/input.mp4") == b"hello video"


def test_large_object_comes_back_in_chunks(storage):
    data = bytes(range(256)) * (CHUNK_SIZE // 256 * 2 + 10)  # just over 2 chunks
    storage.put("big.bin", io.BytesIO(data))
    chunks = list(storage.get("big.bin"))
    assert len(chunks) == 3
    assert b"".join(chunks) == data


def test_put_replaces_an_existing_object(storage):
    storage.put("k", io.BytesIO(b"old"))
    storage.put("k", io.BytesIO(b"new"))
    assert read(storage, "k") == b"new"


def test_get_missing_key_raises_before_iteration(storage):
    # The error must come from get() itself, not from the first next():
    # the API turns it into a 404/410 before it starts a streaming response.
    with pytest.raises(StorageNotFound):
        storage.get("does/not/exist.mp4")


def test_delete_removes_the_object(storage):
    storage.put("k", io.BytesIO(b"x"))
    storage.delete("k")
    with pytest.raises(StorageNotFound):
        storage.get("k")


def test_delete_missing_key_is_not_an_error(storage):
    storage.delete("never/existed")


def test_url_is_none_or_an_http_url(storage):
    storage.put("k", io.BytesIO(b"x"))
    url = storage.url("k")
    assert url is None or url.startswith(("http://", "https://"))


# ---------------------------------------------------------------------------
# Local backend specifics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["../outside.txt", "a/../../outside.txt", "/abs.txt", ""])
def test_local_rejects_keys_that_escape_the_root(tmp_path, key):
    store = LocalStorage(str(tmp_path / "store"))
    with pytest.raises(ValueError):
        store.put(key, io.BytesIO(b"x"))
    assert not (tmp_path / "outside.txt").exists()


def test_local_has_no_download_url(tmp_path):
    assert LocalStorage(str(tmp_path)).url("anything") is None


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_default_backend_is_local(monkeypatch, tmp_path):
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    monkeypatch.setenv("JOBS_DIR", str(tmp_path))
    store = get_storage()
    assert isinstance(store, LocalStorage)
    assert store.root == tmp_path.resolve()


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    with pytest.raises(ValueError, match="STORAGE_BACKEND"):
        get_storage()
