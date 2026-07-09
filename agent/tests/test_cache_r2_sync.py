"""Tests für agent/cache_r2.py — die R2-Backing-Schicht für den lokalen
Datei-Cache (LAUNCH.md P0-3b, Ersatz für den nicht verfügbaren Render
Persistent Disk auf dem Free-Plan). Alle boto3-Aufrufe sind gemockt, kein
Netzwerkzugriff."""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agent.cache_r2 import R2CacheSync

R2_ENV = {
    "R2_ACCOUNT_ID": "acc123",
    "R2_ACCESS_KEY_ID": "key",
    "R2_SECRET_ACCESS_KEY": "secret",
    "R2_BUCKET_NAME": "test-bucket",
}


def _clear_r2_env(monkeypatch):
    for var in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"):
        monkeypatch.delenv(var, raising=False)


def test_disabled_without_env_vars(monkeypatch):
    _clear_r2_env(monkeypatch)
    sync = R2CacheSync()
    assert sync.enabled is False


def test_disabled_with_partial_env_vars(monkeypatch):
    _clear_r2_env(monkeypatch)
    monkeypatch.setenv("R2_ACCOUNT_ID", "acc123")
    monkeypatch.setenv("R2_BUCKET_NAME", "test-bucket")
    sync = R2CacheSync()
    assert sync.enabled is False


def test_warm_and_persist_are_noop_when_disabled(tmp_path, monkeypatch):
    _clear_r2_env(monkeypatch)
    sync = R2CacheSync()
    local_path = tmp_path / "AAPL_stock_financials.json"

    sync.warm(str(local_path), "AAPL_stock_financials.json")
    assert not local_path.exists()

    local_path.write_text("{}")
    sync.persist(str(local_path), "AAPL_stock_financials.json")  # darf nicht raisen


def _patch_env(monkeypatch):
    for key, value in R2_ENV.items():
        monkeypatch.setenv(key, value)


def test_enabled_builds_client_with_r2_endpoint(monkeypatch):
    _patch_env(monkeypatch)
    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client.return_value = MagicMock()
        sync = R2CacheSync()

    assert sync.enabled is True
    _, kwargs = mock_boto3_client.call_args
    assert kwargs["endpoint_url"] == "https://acc123.r2.cloudflarestorage.com"
    assert kwargs["aws_access_key_id"] == "key"
    assert kwargs["aws_secret_access_key"] == "secret"


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
        sync = R2CacheSync()

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
        sync = R2CacheSync()

    local_path = tmp_path / "AAPL_stock_financials.json"
    local_path.write_text("{}")

    sync.warm(str(local_path), "AAPL_stock_financials.json")

    mock_client.head_object.assert_not_called()
    mock_client.download_file.assert_not_called()


def test_warm_swallows_missing_object_in_r2(tmp_path, monkeypatch):
    _patch_env(monkeypatch)
    from botocore.exceptions import ClientError

    mock_client = MagicMock()
    mock_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )

    with patch("boto3.client", return_value=mock_client):
        sync = R2CacheSync()

    local_path = tmp_path / "AAPL_stock_financials.json"
    sync.warm(str(local_path), "AAPL_stock_financials.json")  # darf nicht raisen

    assert not local_path.exists()
    mock_client.download_file.assert_not_called()


def test_persist_uploads_to_r2_when_enabled(tmp_path, monkeypatch):
    _patch_env(monkeypatch)
    mock_client = MagicMock()

    with patch("boto3.client", return_value=mock_client):
        sync = R2CacheSync()

    local_path = tmp_path / "AAPL_stock_financials.json"
    local_path.write_text("{}")

    sync.persist(str(local_path), "AAPL_stock_financials.json")

    mock_client.upload_file.assert_called_once_with(
        str(local_path), "test-bucket", "AAPL_stock_financials.json"
    )


def test_persist_swallows_upload_errors(tmp_path, monkeypatch):
    _patch_env(monkeypatch)
    mock_client = MagicMock()
    mock_client.upload_file.side_effect = ConnectionError("R2 unreachable")

    with patch("boto3.client", return_value=mock_client):
        sync = R2CacheSync()

    local_path = tmp_path / "AAPL_stock_financials.json"
    local_path.write_text("{}")

    sync.persist(str(local_path), "AAPL_stock_financials.json")  # darf nicht raisen
