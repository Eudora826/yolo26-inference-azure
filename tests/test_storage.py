"""Contract tests for the storage backends.

Every backend must pass the same tests: that is what makes them
interchangeable. A new backend joins by adding one entry to the `storage`
fixture's params.
"""

import io
from urllib.parse import urlsplit

import httpx
import pytest

from app.storage import LocalStorage, StorageNotFound, get_storage
from app.storage.base import CHUNK_SIZE


@pytest.fixture(params=["local", "blob"])
def storage(request, tmp_path):
    if request.param == "local":
        return LocalStorage(str(tmp_path / "store"))
    if request.param == "blob":
        # Fixtures first: they skip (or fail in CI) when Azurite is missing.
        connection_string = request.getfixturevalue("azurite")
        container = request.getfixturevalue("blob_container")
        from app.storage.blob import BlobStorage

        return BlobStorage(connection_string, container)
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
# Blob backend specifics (Azurite)
# ---------------------------------------------------------------------------


def test_blob_sas_url_downloads_the_object(azurite, blob_container):
    from app.storage.blob import BlobStorage

    store = BlobStorage(azurite, blob_container)
    store.put("job1/annotated.mp4", io.BytesIO(b"fake video bytes"))
    url = store.url("job1/annotated.mp4", expires_in=60)

    response = httpx.get(url)
    assert response.status_code == 200
    assert response.content == b"fake video bytes"
    # Set at upload time, so the browser treats it as a video download.
    assert response.headers["content-type"] == "video/mp4"
    assert "annotated.mp4" in response.headers["content-disposition"]


def test_blob_sas_url_is_read_only(azurite, blob_container):
    from app.storage.blob import BlobStorage

    store = BlobStorage(azurite, blob_container)
    store.put("k.bin", io.BytesIO(b"original"))
    url = store.url("k.bin", expires_in=60)
    # A leaked download link must not let anyone overwrite the result.
    response = httpx.put(url, content=b"tampered",
                         headers={"x-ms-blob-type": "BlockBlob"})
    assert response.status_code == 403
    assert b"".join(store.get("k.bin")) == b"original"


def test_blob_public_endpoint_rewrites_the_host(azurite, blob_container):
    from app.storage.blob import BlobStorage

    store = BlobStorage(azurite, blob_container,
                        public_endpoint="http://localhost:10000/devstoreaccount1")
    store.put("k.bin", io.BytesIO(b"x"))
    url = store.url("k.bin", expires_in=60)
    assert urlsplit(url).netloc == "localhost:10000"
    # Still a valid link: the signature does not cover the host name.
    assert httpx.get(url).content == b"x"


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_default_backend_is_local(monkeypatch, tmp_path):
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    monkeypatch.setenv("JOBS_DIR", str(tmp_path))
    store = get_storage()
    assert isinstance(store, LocalStorage)
    assert store.root == tmp_path.resolve()


def test_blob_backend_requires_a_connection_string(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "blob")
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING"):
        get_storage()


def test_blob_backend_is_selected_by_env(monkeypatch, azurite, blob_container):
    from app.storage.blob import BlobStorage

    monkeypatch.setenv("STORAGE_BACKEND", "blob")
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", azurite)
    monkeypatch.setenv("STORAGE_CONTAINER", blob_container)
    assert isinstance(get_storage(), BlobStorage)


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    with pytest.raises(ValueError, match="STORAGE_BACKEND"):
        get_storage()
