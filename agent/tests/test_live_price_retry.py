"""Regressionstest für LAUNCH_AUDIT.md P2-12 (`@retry` wirkungslos, weil die
Methode intern jede Exception fängt und ein error-dict zurückgibt, statt sie
propagieren zu lassen - der Decorator sieht dann nie einen Fehler und
retried nie) sowie für den Alpha-Vantage-Fallback aus LAUNCH.md P2-22 (Yahoo
blockt/rate-limitet Cloud-IPs wiederkehrend, nicht nur transient - ein
Retry allein reicht nicht, es braucht eine zweite Quelle).

`get_current_price_per_share` (agent/DataLoader.py) war live in Produktion
betroffen: Yahoo/yfinance blockt/rate-limitet Anfragen von Cloud-IPs (Render
etc.) häufiger als von privaten IPs, ein einzelner 429/Timeout wurde bisher
sofort zum permanenten Fehler statt erneut versucht zu werden.

Kein Netzwerkzugriff: `yf.Ticker` und `requests.get` werden mit
unittest.mock.patch ersetzt.
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


def test_get_current_price_per_share_raises_after_exhausting_retries_and_fallback(loader):
    """Dauerhafter Fehler (z. B. Yahoo blockt die IP komplett) UND ein
    ebenfalls scheiternder Alpha-Vantage-Fallback müssen nach den
    konfigurierten 3 Yahoo-Versuchen als Exception durchschlagen, nicht als
    stillschweigendes error-dict - alle Aufrufer fangen das in einem
    eigenen try/except ab (siehe LAUNCH.md Retry-Fix-Notiz). tenacity ohne
    reraise=True wrappt die letzte Exception in RetryError statt sie 1:1
    weiterzureichen - die ursprüngliche ConnectionError bleibt als
    `__cause__` erhalten (sichtbar in Log-Tracebacks via logger.exception)."""
    with patch("agent.DataLoader.yf.Ticker", side_effect=ConnectionError("429 Too Many Requests")) as mock_ticker_cls, \
            patch("agent.DataLoader.requests.get", side_effect=ConnectionError("AV unreachable")) as mock_av_get:
        with pytest.raises(RetryError) as exc_info:
            loader.get_current_price_per_share("TSLA")

    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert mock_ticker_cls.call_count == 3
    mock_av_get.assert_called_once()


def test_get_current_price_per_share_does_not_call_alpha_vantage_when_yahoo_succeeds(loader):
    fake_ticker = MagicMock()
    fake_ticker.info = {"regularMarketPrice": 123.45}

    with patch("agent.DataLoader.yf.Ticker", return_value=fake_ticker), \
            patch("agent.DataLoader.requests.get") as mock_av_get:
        price = loader.get_current_price_per_share("AAPL")

    assert price == 123.45
    mock_av_get.assert_not_called()


def test_get_current_price_per_share_falls_back_to_alpha_vantage_delayed_tier(loader):
    """Live gegen die echte API verifiziert (2026-07-10): der bezahlte
    "15-Minuten-verzögert"-Tarif (entitlement=delayed, den dieser Account
    hat) liefert den Preis unter "Global Quote - DATA DELAYED BY 15
    MINUTES", NICHT unter "Global Quote" - ein hartkodierter exakter Key
    hätte hier still None zurückgegeben und den RetryError re-raised, obwohl
    der Fallback eigentlich einen validen Preis hatte (live bei MO/PYPL
    beobachtet: KGV/KUV schlugen mit RetryError fehl, obwohl Alpha Vantage
    einen Preis geliefert hätte)."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "Global Quote - DATA DELAYED BY 15 MINUTES": {"05. price": "88.20"}
    }

    with patch("agent.DataLoader.yf.Ticker", side_effect=ConnectionError("429 Too Many Requests")) as mock_ticker_cls, \
            patch("agent.DataLoader.requests.get", return_value=mock_response) as mock_av_get:
        price = loader.get_current_price_per_share("TSLA")

    assert price == 88.20
    assert mock_ticker_cls.call_count == 3
    mock_av_get.assert_called_once()
    _, kwargs = mock_av_get.call_args
    assert kwargs["params"]["function"] == "GLOBAL_QUOTE"
    assert kwargs["params"]["entitlement"] == "delayed"


def test_get_current_price_per_share_falls_back_to_alpha_vantage_realtime_tier(loader):
    """Der kostenlose/Realtime-Tarif liefert den Preis unter dem einfachen
    Key "Global Quote" - muss weiterhin funktionieren, falls der Account
    später auf einen anderen Tarif wechselt."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"Global Quote": {"05. price": "42.00"}}

    with patch("agent.DataLoader.yf.Ticker", side_effect=ConnectionError("429 Too Many Requests")), \
            patch("agent.DataLoader.requests.get", return_value=mock_response):
        price = loader.get_current_price_per_share("TSLA")

    assert price == 42.00


def test_get_current_price_per_share_raises_when_alpha_vantage_rate_limited(loader):
    """Alpha-Vantage-Rate-Limit-Antworten kommen als HTTP 200 mit einem
    "Note"/"Information"-Feld, nicht als HTTP-Fehler - muss trotzdem als
    Fallback-Fehlschlag erkannt werden, nicht als valider Preis."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"Information": "rate limit exceeded"}

    with patch("agent.DataLoader.yf.Ticker", side_effect=ConnectionError("429 Too Many Requests")), \
            patch("agent.DataLoader.requests.get", return_value=mock_response):
        with pytest.raises(RetryError) as exc_info:
            loader.get_current_price_per_share("TSLA")

    assert isinstance(exc_info.value.__cause__, ConnectionError)
