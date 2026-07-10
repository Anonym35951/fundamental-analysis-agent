"""Integrationstests: DataLoader/SecSource rufen den Object-Storage-Sync an
den richtigen Stellen mit den richtigen Keys auf (LAUNCH.md P0-3b)."""

from unittest.mock import MagicMock

from agent.DataLoader import DataLoader
from agent.data_sources.sec_source import SecSource


def test_dataloader_cache_data_persists_to_object_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    monkeypatch.delenv("CACHE_S3_ENDPOINT_URL", raising=False)

    loader = DataLoader()
    loader._cache_sync = MagicMock()

    loader._cache_data({"value": 1}, "AAPL", "some_metric")

    loader._cache_sync.persist.assert_called_once()
    args, _ = loader._cache_sync.persist.call_args
    assert args[1] == "AAPL_some_metric.json"


def test_dataloader_load_cached_data_warms_from_object_storage_on_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    monkeypatch.delenv("CACHE_S3_ENDPOINT_URL", raising=False)

    loader = DataLoader()
    loader._cache_sync = MagicMock()

    result = loader._load_cached_data("AAPL", "some_metric")

    loader._cache_sync.warm.assert_called_once()
    args, _ = loader._cache_sync.warm.call_args
    assert args[1] == "AAPL_some_metric.json"
    assert result is None  # Mock legt keine Datei an -> weiterhin Cache-Miss


def test_sec_source_cache_data_persists_with_sec_prefix(tmp_path):
    source = SecSource(user_agent="test@example.com", cache_dir=str(tmp_path))
    source._cache_sync = MagicMock()

    source._cache_data({"value": 1}, "AAPL_balance_sheet")

    source._cache_sync.persist.assert_called_once()
    args, _ = source._cache_sync.persist.call_args
    assert args[1] == "sec/AAPL_balance_sheet.json"


def test_sec_source_load_cached_data_warms_with_sec_prefix(tmp_path):
    source = SecSource(user_agent="test@example.com", cache_dir=str(tmp_path))
    source._cache_sync = MagicMock()

    result = source._load_cached_data("AAPL_balance_sheet")

    source._cache_sync.warm.assert_called_once()
    args, _ = source._cache_sync.warm.call_args
    assert args[1] == "sec/AAPL_balance_sheet.json"
    assert result is None
