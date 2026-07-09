"""Regressionstest für LAUNCH_AUDIT.md P2-12 (`@retry` wirkungslos, weil die
Methode intern jede Exception fängt und ein error-dict zurückgibt, statt sie
propagieren zu lassen - der Decorator sieht dann nie einen Fehler und
retried nie).

`get_current_price_per_share` (agent/DataLoader.py) war live in Produktion
betroffen: Yahoo/yfinance blockt/rate-limitet Anfragen von Cloud-IPs (Render
etc.) häufiger als von privaten IPs, ein einzelner 429/Timeout wurde bisher
sofort zum permanenten Fehler statt erneut versucht zu werden.

Kein Netzwerkzugriff: `yf.Ticker` wird mit unittest.mock.patch ersetzt.
"""
import pytest
from tenacity import RetryError
from unittest.mock import MagicMock, patch

from agent.DataLoader import DataLoader


@pytest.fixture
def loader():
    return DataLoader()


def test_get_current_price_per_share_succeeds_without_retry(loader):
    fake_ticker = MagicMock()
    fake_ticker.info = {"regularMarketPrice": 123.45}

    with patch("agent.DataLoader.yf.Ticker", return_value=fake_ticker) as mock_ticker_cls:
        price = loader.get_current_price_per_share("AAPL")

    assert price == 123.45
    assert mock_ticker_cls.call_count == 1


def test_get_current_price_per_share_retries_on_transient_failure(loader):
    """Zwei fehlgeschlagene Versuche (z. B. Yahoo-429), dritter erfolgreich -
    beweist, dass @retry jetzt tatsächlich mehrfach versucht statt beim
    ersten Fehler ein error-dict zurückzugeben."""
    fake_ticker = MagicMock()
    fake_ticker.info = {"regularMarketPrice": 55.0}

    call_count = {"n": 0}

    def flaky_ticker(symbol):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("429 Too Many Requests")
        return fake_ticker

    with patch("agent.DataLoader.yf.Ticker", side_effect=flaky_ticker):
        price = loader.get_current_price_per_share("MSFT")

    assert price == 55.0
    assert call_count["n"] == 3


def test_get_current_price_per_share_raises_after_exhausting_retries(loader):
    """Dauerhafter Fehler (z. B. Yahoo blockt die IP komplett) muss nach den
    konfigurierten 3 Versuchen als Exception durchschlagen, nicht als
    stillschweigendes error-dict - alle Aufrufer fangen das in einem
    eigenen try/except ab (siehe LAUNCH.md Retry-Fix-Notiz). tenacity ohne
    reraise=True wrappt die letzte Exception in RetryError statt sie 1:1
    weiterzureichen - die ursprüngliche ConnectionError bleibt als
    `__cause__` erhalten (sichtbar in Log-Tracebacks via logger.exception)."""
    with patch("agent.DataLoader.yf.Ticker", side_effect=ConnectionError("429 Too Many Requests")) as mock_ticker_cls:
        with pytest.raises(RetryError) as exc_info:
            loader.get_current_price_per_share("TSLA")

    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert mock_ticker_cls.call_count == 3
