"""Integrationstests: DataLoader/SecSource rufen den R2-Sync an den
richtigen Stellen mit den richtigen Keys auf (LAUNCH.md P0-3b)."""

from unittest.mock import MagicMock, patch

from agent.DataLoader import DataLoader
from agent.data_sources.sec_source import SecSource


def test_dataloader_cache_data_persists_to_r2(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)

    loader = DataLoader()
    loader._r2 = MagicMock()

    loader._cache_data({"value": 1}, "AAPL", "some_metric")

    loader._r2.persist.assert_called_once()
    args, _ = loader._r2.persist.call_args
    assert args[1] == "AAPL_some_metric.json"


def test_dataloader_load_cached_data_warms_from_r2_on_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)

    loader = DataLoader()
    loader._r2 = MagicMock()

    result = loader._load_cached_data("AAPL", "some_metric")

    loader._r2.warm.assert_called_once()
    args, _ = loader._r2.warm.call_args
    assert args[1] == "AAPL_some_metric.json"
    assert result is None  # R2-Mock legt keine Datei an -> weiterhin Cache-Miss


def test_sec_source_cache_data_persists_with_sec_prefix(tmp_path):
    source = SecSource(user_agent="test@example.com", cache_dir=str(tmp_path))
    source._r2 = MagicMock()

    source._cache_data({"value": 1}, "AAPL_balance_sheet")

    source._r2.persist.assert_called_once()
    args, _ = source._r2.persist.call_args
    assert args[1] == "sec/AAPL_balance_sheet.json"


def test_sec_source_load_cached_data_warms_with_sec_prefix(tmp_path):
    source = SecSource(user_agent="test@example.com", cache_dir=str(tmp_path))
    source._r2 = MagicMock()

    result = source._load_cached_data("AAPL_balance_sheet")

    source._r2.warm.assert_called_once()
    args, _ = source._r2.warm.call_args
    assert args[1] == "sec/AAPL_balance_sheet.json"
    assert result is None
