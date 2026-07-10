"""Tests für agent/cache_object_storage.py — die Object-Storage-Backing-
Schicht für den lokalen Datei-Cache (LAUNCH.md P0-3b, Ersatz für den nicht
verfügbaren Render Persistent Disk auf dem Free-Plan; konfiguriert für
Backblaze B2). Alle boto3-Aufrufe sind gemockt, kein Netzwerkzugriff."""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agent.cache_object_storage import ObjectStorageCacheSync

CACHE_S3_ENV = {
    "CACHE_S3_ENDPOINT_URL": "s3.us-west-004.backblazeb2.com",
    "CACHE_S3_ACCESS_KEY_ID": "key",
    "CACHE_S3_SECRET_ACCESS_KEY": "secret",
    "CACHE_S3_BUCKET_NAME": "test-bucket",
}


def _clear_env(monkeypatch):
    for var in CACHE_S3_ENV:
        monkeypatch.delenv(var, raising=False)


def _patch_env(monkeypatch):
    for key, value in CACHE_S3_ENV.items():
        monkeypatch.setenv(key, value)


def test_disabled_without_env_vars(monkeypatch):
    _clear_env(monkeypatch)
    sync = ObjectStorageCacheSync()
    assert sync.enabled is False


def test_disabled_with_partial_env_vars(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("CACHE_S3_ENDPOINT_URL", "s3.us-west-004.backblazeb2.com")
    monkeypatch.setenv("CACHE_S3_BUCKET_NAME", "test-bucket")
    sync = ObjectStorageCacheSync()
    assert sync.enabled is False


def test_warm_and_persist_are_noop_when_disabled(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    sync = ObjectStorageCacheSync()
    local_path = tmp_path / "AAPL_stock_financials.json"

    sync.warm(str(local_path), "AAPL_stock_financials.json")
    assert not local_path.exists()

    local_path.write_text("{}")
    sync.persist(str(local_path), "AAPL_stock_financials.json")  # darf nicht raisen


def test_enabled_builds_client_with_normalized_endpoint_and_derived_region(monkeypatch):
    _patch_env(monkeypatch)
    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client.return_value = MagicMock()
        sync = ObjectStorageCacheSync()

    assert sync.enabled is True
    _, kwargs = mock_boto3_client.call_args
    assert kwargs["endpoint_url"] == "https://s3.us-west-004.backblazeb2.com"
    assert kwargs["aws_access_key_id"] == "key"
    assert kwargs["aws_secret_access_key"] == "secret"
    assert kwargs["region_name"] == "us-west-004"


def test_endpoint_with_scheme_already_present_is_left_unchanged(monkeypatch):
    _clear_env(monkeypatch)
    for key, value in CACHE_S3_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CACHE_S3_ENDPOINT_URL", "https://s3.eu-central-003.backblazeb2.com")

    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client.return_value = MagicMock()
        ObjectStorageCacheSync()

    _, kwargs = mock_boto3_client.call_args
    assert kwargs["endpoint_url"] == "https://s3.eu-central-003.backblazeb2.com"
    assert kwargs["region_name"] == "eu-central-003"


def test_warm_downloads_and_sets_mtime_when_local_missing(tmp_path, monkeypatch):
    _patch_env(monkeypatch)
    last_modified = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mock_client = MagicMock()
    mock_client.head_object.return_value = {"LastModified": last_modified}

    def _fake_download(bucket, key, local_path):
        with open(local_path, "w") as f:
            f.write('{"cached": true}')

    mock_client.download_file.side_effect = _fake_download

    with patch("boto3.client", return_value=mock_client):
        sync = ObjectStorageCacheSync()

    local_path = tmp_path / "AAPL_stock_financials.json"
    sync.warm(str(local_path), "AAPL_stock_financials.json")

    assert local_path.exists()
    mock_client.head_object.assert_called_once_with(Bucket="test-bucket", Key="AAPL_stock_financials.json")
    mock_client.download_file.assert_called_once_with(
        "test-bucket", "AAPL_stock_financials.json", str(local_path)
    )
    assert os.path.getmtime(local_path) == pytest.approx(last_modified.timestamp(), abs=1)


def test_warm_skips_download_when_local_file_exists(tmp_path, monkeypatch):
    _patch_env(monkeypatch)
    mock_client = MagicMock()

    with patch("boto3.client", return_value=mock_client):
        sync = ObjectStorageCacheSync()

    local_path = tmp_path / "AAPL_stock_financials.json"
    local_path.write_text("{}")

    sync.warm(str(local_path), "AAPL_stock_financials.json")

    mock_client.head_object.assert_not_called()
    mock_client.download_file.assert_not_called()


def test_warm_swallows_missing_object(tmp_path, monkeypatch):
    _patch_env(monkeypatch)
    from botocore.exceptions import ClientError

    mock_client = MagicMock()
    mock_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )

    with patch("boto3.client", return_value=mock_client):
        sync = ObjectStorageCacheSync()

    local_path = tmp_path / "AAPL_stock_financials.json"
    sync.warm(str(local_path), "AAPL_stock_financials.json")  # darf nicht raisen

    assert not local_path.exists()
    mock_client.download_file.assert_not_called()


def test_persist_uploads_when_enabled(tmp_path, monkeypatch):
    _patch_env(monkeypatch)
    mock_client = MagicMock()

    with patch("boto3.client", return_value=mock_client):
        sync = ObjectStorageCacheSync()

    local_path = tmp_path / "AAPL_stock_financials.json"
    local_path.write_text("{}")

    sync.persist(str(local_path), "AAPL_stock_financials.json")

    mock_client.upload_file.assert_called_once_with(
        str(local_path), "test-bucket", "AAPL_stock_financials.json"
    )


def test_persist_swallows_upload_errors(tmp_path, monkeypatch):
    _patch_env(monkeypatch)
    mock_client = MagicMock()
    mock_client.upload_file.side_effect = ConnectionError("storage unreachable")

    with patch("boto3.client", return_value=mock_client):
        sync = ObjectStorageCacheSync()

    local_path = tmp_path / "AAPL_stock_financials.json"
    local_path.write_text("{}")

    sync.persist(str(local_path), "AAPL_stock_financials.json")  # darf nicht raisen
