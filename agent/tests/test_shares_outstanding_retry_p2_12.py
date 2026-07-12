"""Regressionstest für LAUNCH_AUDIT.md P2-12: DataLoader.get_shares_outstanding
fing den kompletten Yahoo-Fallback-Zweig in einem umschließenden try/except
ab, wodurch @retry nie einen echten Fehler sah. Der Yahoo-Call ist jetzt in
_fetch_yahoo_shares_outstanding isoliert (mit eigenem @retry) - der äußere
Vertrag (immer ein Dict, nie eine Exception) bleibt für get_shares_outstanding
unverändert.

Kein Netzwerkzugriff: yf.Ticker und der SEC-Pfad werden gemockt.
"""
from unittest.mock import MagicMock, patch

import pytest
from tenacity import RetryError

from agent.DataLoader import DataLoader


@pytest.fixture
def loader(tmp_path):
    dl = DataLoader()
    dl.cache_dir = str(tmp_path)
    return dl


def _force_sec_path_unavailable(loader):
    """Lässt den SEC-Pfad (bewusst try/except: pass) durchfallen, damit der
    Test isoliert den Yahoo-Fallback prüft."""
    return (
        patch.object(loader.sec_source, "is_foreign_private_issuer", return_value=False),
        patch.object(
            loader.sec_source,
            "get_balance_sheet_line_item",
            side_effect=Exception("SEC nicht verfügbar"),
        ),
    )


def test_get_shares_outstanding_yahoo_fallback_succeeds_without_retry(loader):
    fake_ticker = MagicMock()
    fake_ticker.info = {"sharesOutstanding": 1_000_000}
    p1, p2 = _force_sec_path_unavailable(loader)

    with p1, p2, patch("agent.DataLoader.yf.Ticker", return_value=fake_ticker) as mock_ticker_cls:
        result = loader.get_shares_outstanding("AAPL")

    assert result["shares_outstanding"] == 1_000_000.0
    assert result["source"] == "Yahoo Finance"
    assert mock_ticker_cls.call_count == 1


def test_get_shares_outstanding_yahoo_fallback_retries_on_transient_failure(loader):
    fake_ticker = MagicMock()
    fake_ticker.info = {"sharesOutstanding": 2_000_000}
    call_count = {"n": 0}

    def flaky_ticker(symbol):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("transient network error")
        return fake_ticker

    p1, p2 = _force_sec_path_unavailable(loader)
    with p1, p2, patch("agent.DataLoader.yf.Ticker", side_effect=flaky_ticker):
        result = loader.get_shares_outstanding("MSFT")

    assert result["shares_outstanding"] == 2_000_000.0
    assert call_count["n"] == 3


def test_get_shares_outstanding_returns_error_dict_after_exhausted_retries(loader):
    """Äußerer Vertrag bleibt unverändert: nach erschöpften Retries ein
    sauberes {"error": ...}-Dict, kein rohes RetryError - get_shares_outstanding
    hat weiterhin sein eigenes try/except um den Helper."""
    p1, p2 = _force_sec_path_unavailable(loader)
    with p1, p2, patch("agent.DataLoader.yf.Ticker", side_effect=ConnectionError("down")):
        result = loader.get_shares_outstanding("TSLA")

    assert isinstance(result, dict)
    assert "error" in result


def test_fetch_yahoo_shares_outstanding_raises_retry_error_directly(loader):
    """Der isolierte Helper selbst propagiert nach erschöpften Retries eine
    Exception (kein Dict) - genau das macht ihn für @retry sichtbar."""
    with patch("agent.DataLoader.yf.Ticker", side_effect=ConnectionError("down")):
        with pytest.raises(RetryError):
            loader._fetch_yahoo_shares_outstanding("TSLA")
