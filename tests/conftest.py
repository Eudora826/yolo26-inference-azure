"""Shared test helpers: access to Azurite for the Blob storage tests."""

import os
import socket
import uuid

import pytest

# The well-known Azurite development account, published by Microsoft. Only
# valid for the emulator; not a secret.
AZURITE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
    "K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


def _azurite_running() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 10000), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture()
def azurite():
    """Skip when Azurite is not running locally; fail instead in CI, where
    REQUIRE_AZURITE=1 means the Blob tests must actually run."""
    if not _azurite_running():
        message = "Azurite is not listening on 127.0.0.1:10000"
        if os.getenv("REQUIRE_AZURITE") == "1":
            pytest.fail(message + " (REQUIRE_AZURITE=1)")
        pytest.skip(message + "; start it to run the Blob tests")
    return AZURITE_CONNECTION_STRING


@pytest.fixture()
def blob_container(azurite):
    """A fresh container per test, deleted afterwards, so tests never see
    each other's blobs."""
    from azure.storage.blob import BlobServiceClient

    name = "test-" + uuid.uuid4().hex[:12]
    yield name
    service = BlobServiceClient.from_connection_string(azurite)
    try:
        service.delete_container(name)
    except Exception:
        pass
