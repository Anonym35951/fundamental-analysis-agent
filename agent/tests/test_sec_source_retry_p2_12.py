"""Regressionstests für LAUNCH_AUDIT.md P2-12: sec_source.py::get_cik und
get_company_facts sind der Choke-Point hinter praktisch jedem SEC-Datenpfad
(get_balance_sheet, get_stock_financials, get_cashflow_statement usw. rufen
alle transitiv hierher durch, 19+9+ Aufrufstellen). @retry war wirkungslos,
weil ein umschließendes try/except JEDE Exception (auch transiente
Netzwerkfehler auf dem echten requests.get) abfing, bevor der Decorator sie
je sehen konnte.

Kein Netzwerkzugriff: requests.get wird gemockt. tmp_path als Cache-Dir,
damit Tests nicht mit dem echten Cache interferieren.
"""
from unittest.mock import MagicMock, patch

import pytest
from tenacity import RetryError

from agent.data_sources.sec_source import SecSource


@pytest.fixture
def sec(tmp_path):
    return SecSource(user_agent="test@example.com", cache_dir=str(tmp_path))


def _fake_response(json_data):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = json_data
    return response


def test_get_cik_succeeds_without_retry(sec):
    tickers = {"0": {"ticker": "AAPL", "cik_str": 320193}}

    with patch("agent.data_sources.sec_source.requests.get", return_value=_fake_response(tickers)) as mock_get:
        cik = sec.get_cik("AAPL")

    assert cik == "0000320193"
    assert mock_get.call_count == 1


def test_get_cik_retries_on_transient_failure(sec):
    """Zwei fehlgeschlagene Versuche, dritter erfolgreich - beweist, dass
    @retry jetzt tatsächlich mehrfach versucht statt beim ersten
    Netzwerkfehler sofort ein error-dict zurückzugeben."""
    tickers = {"0": {"ticker": "AAPL", "cik_str": 320193}}
    call_count = {"n": 0}

    def flaky_get(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("transient network error")
        return _fake_response(tickers)

    with patch("agent.data_sources.sec_source.requests.get", side_effect=flaky_get):
        cik = sec.get_cik("AAPL")

    assert cik == "0000320193"
    assert call_count["n"] == 3


def test_get_cik_gives_up_after_exhausted_retries(sec):
    with patch("agent.data_sources.sec_source.requests.get", side_effect=ConnectionError("down")):
        with pytest.raises(RetryError):
            sec.get_cik("AAPL")


def test_get_cik_unknown_ticker_returns_dict_not_exception(sec):
    """Ein Symbol, das schlicht nicht in der Liste steht, ist kein
    transienter Fehler - kein Retry sinnvoll, weiterhin ein Dict."""
    tickers = {"0": {"ticker": "AAPL", "cik_str": 320193}}

    with patch("agent.data_sources.sec_source.requests.get", return_value=_fake_response(tickers)) as mock_get:
        result = sec.get_cik("NOTREAL")

    assert isinstance(result, dict)
    assert "error" in result
    assert mock_get.call_count == 1  # kein Retry auf eine legitime "nicht gefunden"-Antwort


def test_get_company_facts_retries_on_transient_failure(sec):
    """Isoliert get_cik (Erfolg beim ersten Versuch) vom eigentlichen
    Company-Facts-Call, der hier die transienten Fehler simuliert."""
    facts_payload = {"facts": {"us-gaap": {}}}
    call_count = {"n": 0}

    def flaky_get(url, *args, **kwargs):
        if "company_tickers" in url:
            return _fake_response({"0": {"ticker": "AAPL", "cik_str": 320193}})
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("transient network error")
        return _fake_response(facts_payload)

    with patch("agent.data_sources.sec_source.requests.get", side_effect=flaky_get):
        facts = sec.get_company_facts("AAPL")

    assert facts == facts_payload
    assert call_count["n"] == 3


def test_get_company_facts_propagates_cik_not_found_without_retry(sec):
    with patch(
        "agent.data_sources.sec_source.requests.get",
        return_value=_fake_response({"0": {"ticker": "AAPL", "cik_str": 320193}}),
    ) as mock_get:
        result = sec.get_company_facts("NOTREAL")

    assert isinstance(result, dict)
    assert "error" in result
    # Nur der Ticker-Listen-Call, kein zweiter Versuch fuer die (nie
    # erreichte) Company-Facts-URL.
    assert mock_get.call_count == 1


def test_get_balance_sheet_still_returns_error_dict_after_exhausted_retries(sec):
    """Der äußere Vertrag bleibt unverändert: get_balance_sheet (ein
    Aufrufer von get_company_facts) liefert nach erschöpften Retries
    weiterhin ein sauberes {"error": ...}-Dict, kein rohes RetryError -
    P2-12 sollte NUR den Retry selbst wirksam machen, nicht den bestehenden
    Fehler-Contract für Aufrufer brechen."""
    with patch("agent.data_sources.sec_source.requests.get", side_effect=ConnectionError("down")):
        result = sec.get_balance_sheet("AAPL", frequency="annual", use_cache=False)

    assert isinstance(result, dict)
    assert "error" in result
