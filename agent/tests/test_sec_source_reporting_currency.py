"""Regressionstests für EVOLVING.md EV-020: SecSource.get_reporting_currency
ermittelt die ISO-Berichtswährung eines Filers aus den bereits vorhandenen
XBRL-Fakten (get_company_facts), ohne die bestehende Kennzahlen-Berechnung
(_merge_available_series & Co., fest codiertes unit_preference=["USD",...])
anzufassen.

Fixtures spiegeln real gegen die SEC-API verifizierte Strukturen wider
(2026-07-14, siehe EVOLVING.md EV-020 "Noch zu verifizieren"):
- AAPL: us-gaap, nur USD-Unit.
- BABA/JD: us-gaap, ZWEI Units (CNY und USD) pro Tag.
- SAP/Novo Nordisk: AUSSCHLIESSLICH ifrs-full, kein us-gaap-Key überhaupt -
  die bestehende Pipeline liefert für solche Filer heute schon keine
  Kennzahlen (liest nur "us-gaap"), get_reporting_currency gibt daher
  bewusst None zurück statt eine Auswertung vorzutäuschen, die der Rest der
  Pipeline nicht leistet.

Kein Netzwerkzugriff: get_company_facts wird direkt gemockt (Stil wie
test_sec_source_retry_p2_12.py, aber ohne den Umweg über requests.get, da
diese Methode nur get_company_facts konsumiert).
"""
from unittest.mock import patch

import pytest

from agent.data_sources.sec_source import SecSource


@pytest.fixture
def sec(tmp_path):
    return SecSource(user_agent="test@example.com", cache_dir=str(tmp_path))


def _facts(us_gaap: dict | None = None, ifrs_full: dict | None = None) -> dict:
    facts: dict = {}
    if us_gaap is not None:
        facts["us-gaap"] = us_gaap
    if ifrs_full is not None:
        facts["ifrs-full"] = ifrs_full
    return {"facts": facts}


def test_us_only_dollar_filer_returns_usd(sec):
    """AAPL-artige Struktur: us-gaap, ausschließlich USD-Unit."""
    facts = _facts(us_gaap={"Assets": {"units": {"USD": [{"val": 1}]}}})

    with patch.object(sec, "get_company_facts", return_value=facts) as mock_facts:
        result = sec.get_reporting_currency("AAPL")

    assert result == "USD"
    mock_facts.assert_called_once_with("AAPL", use_cache=True)


def test_dual_currency_filer_prefers_usd(sec):
    """BABA/JD-artige Struktur: derselbe Tag hat sowohl CNY- als auch
    USD-Unit - die bestehende Pipeline (unit_preference=["USD",...]) waehlt
    hier bereits heute USD, get_reporting_currency muss das widerspiegeln."""
    facts = _facts(
        us_gaap={
            "Assets": {"units": {"CNY": [{"val": 1}], "USD": [{"val": 1}]}},
        }
    )

    with patch.object(sec, "get_company_facts", return_value=facts):
        result = sec.get_reporting_currency("BABA")

    assert result == "USD"


def test_ifrs_full_only_filer_returns_none(sec):
    """SAP/Novo-Nordisk-artige Struktur: gar kein us-gaap-Key. Die bestehende
    Pipeline liest ausschließlich us-gaap und liefert für solche Filer schon
    heute keine Kennzahlen - get_reporting_currency gibt konsequenterweise
    None zurück statt ifrs-full selbst auszuwerten (außerhalb des
    Aufgabenumfangs von EV-020, siehe Docstring der Methode)."""
    facts = _facts(ifrs_full={"Assets": {"units": {"EUR": [{"val": 1}], "USD": [{"val": 1}]}}})

    with patch.object(sec, "get_company_facts", return_value=facts):
        result = sec.get_reporting_currency("SAP")

    assert result is None


def test_us_gaap_present_but_empty_returns_none(sec):
    facts = _facts(us_gaap={})

    with patch.object(sec, "get_company_facts", return_value=facts):
        result = sec.get_reporting_currency("SOMESYM")

    assert result is None


def test_non_usd_only_us_gaap_filer_returns_that_currency(sec):
    """Hypothetischer Fall (in der Stichprobe nicht beobachtet, aber
    technisch möglich): us-gaap-Filer ohne jede USD-Unit auf den
    Sondierungs-Tags - erster reiner 3-Buchstaben-Währungscode wird
    zurückgegeben, nicht geraten."""
    facts = _facts(us_gaap={"Assets": {"units": {"EUR": [{"val": 1}]}}})

    with patch.object(sec, "get_company_facts", return_value=facts):
        result = sec.get_reporting_currency("HYPOCO")

    assert result == "EUR"


def test_composite_units_are_ignored_when_searching_for_currency(sec):
    """"shares"/"pure"/zusammengesetzte Einheiten dürfen nicht fälschlich
    als Währungscode zurückgegeben werden."""
    facts = _facts(us_gaap={"Assets": {"units": {"shares": [{"val": 1}], "pure": [{"val": 1}]}}})

    with patch.object(sec, "get_company_facts", return_value=facts):
        result = sec.get_reporting_currency("NOCURRENCY")

    assert result is None


def test_falls_through_probe_tags_until_one_has_units(sec):
    facts = _facts(
        us_gaap={
            "Assets": {"units": {}},  # vorhanden, aber keine Units
            "Revenues": {"units": {"USD": [{"val": 1}]}},
        }
    )

    with patch.object(sec, "get_company_facts", return_value=facts):
        result = sec.get_reporting_currency("AAPL")

    assert result == "USD"


def test_unknown_symbol_error_dict_returns_none(sec):
    with patch.object(sec, "get_company_facts", return_value={"error": "Kein SEC-CIK gefunden", "symbol": "FOOBARX"}):
        result = sec.get_reporting_currency("FOOBARX")

    assert result is None


def test_use_cache_flag_is_forwarded(sec):
    facts = _facts(us_gaap={"Assets": {"units": {"USD": [{"val": 1}]}}})

    with patch.object(sec, "get_company_facts", return_value=facts) as mock_facts:
        sec.get_reporting_currency("AAPL", use_cache=False)

    mock_facts.assert_called_once_with("AAPL", use_cache=False)
