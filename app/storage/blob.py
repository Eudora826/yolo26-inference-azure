"""Storage in Azure Blob, or in Azurite (Microsoft's local Blob emulator).

The SDK is imported here and this module is only imported when
STORAGE_BACKEND=blob, so the local backend works without azure-storage-blob.
"""

import mimetypes
from datetime import UTC, datetime, timedelta
from typing import BinaryIO, Iterator, Optional
from urllib.parse import urlsplit, urlunsplit

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)

from app.storage.base import CHUNK_SIZE, StorageNotFound


class BlobStorage:
    def __init__(
        self,
        connection_string: str,
        container: str,
        public_endpoint: Optional[str] = None,
    ):
        """
        public_endpoint: where clients reach Blob, if that differs from where
        this service reaches it. Under Docker Compose the service talks to
        http://azurite:10000/devstoreaccount1, a name only containers can
        resolve, while a browser on the host needs
        http://127.0.0.1:10000/devstoreaccount1. Leave it unset on Azure,
        where both are the same public address.
        """
        self._service = BlobServiceClient.from_connection_string(
            connection_string, max_single_get_size=CHUNK_SIZE,
            max_chunk_get_size=CHUNK_SIZE,
        )
        self._container = self._service.get_container_client(container)
        self._public_endpoint = public_endpoint.rstrip("/") if public_endpoint else None
        try:
            self._container.create_container()
        except ResourceExistsError:
            pass

    def put(self, key: str, data: BinaryIO) -> None:
        content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
        self._container.upload_blob(
            name=key,
            data=data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )

    def get(self, key: str) -> Iterator[bytes]:
        try:
            # download_blob() makes the first request immediately, so a
            # missing blob fails here and not on the first iteration.
            downloader = self._container.download_blob(key)
        except ResourceNotFoundError:
            raise StorageNotFound(key) from None
        return downloader.chunks()

    def url(self, key: str, expires_in: int = 900) -> Optional[str]:
        """A read-only SAS URL for one blob, valid for expires_in seconds.

        The client downloads straight from Blob, so a large video never passes
        through the service's own (small) container. Signed with the account
        key from the connection string; a production setup would rather use a
        user delegation key from a managed identity.
        """
        blob = self._container.get_blob_client(key)
        filename = key.rsplit("/", 1)[-1]
        sas = generate_blob_sas(
            account_name=self._service.account_name,
            container_name=self._container.container_name,
            blob_name=key,
            account_key=self._service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(UTC) + timedelta(seconds=expires_in),
            # Overrides the response header, so the browser saves the file
            # under its own name instead of playing or renaming it.
            content_disposition='attachment; filename="{}"'.format(filename),
        )
        return "{}?{}".format(self._public_url(blob.url), sas)

    def delete(self, key: str) -> None:
        try:
            self._container.delete_blob(key)
        except ResourceNotFoundError:
            pass

    def _public_url(self, url: str) -> str:
        """Swap the internal endpoint for the public one. The SAS signature
        covers the account, container and blob name, not the host, so the
        rewritten URL stays valid."""
        if not self._public_endpoint:
            return url
        internal = self._service.url.rstrip("/")
        if not url.startswith(internal):
            return url
        public = urlsplit(self._public_endpoint)
        rest = urlsplit(url[len(internal):] or "/")
        return urlunsplit((public.scheme, public.netloc,
                           public.path.rstrip("/") + rest.path, rest.query, ""))
