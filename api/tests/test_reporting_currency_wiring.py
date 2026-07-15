"""Regressionstests für EVOLVING.md EV-021: resolve_reporting_currency
(api/utils/reporting_currency.py) darf einen laufenden Analyse-Job niemals
zum Absturz bringen, auch wenn die SEC-Currency-Ermittlung selbst
fehlschlägt (Netzwerkfehler, erschöpfte Retries in SecSource)."""
from unittest.mock import MagicMock

from api.utils.reporting_currency import resolve_reporting_currency


def _fake_action(currency_result=None, side_effect=None):
    action = MagicMock()
    if side_effect is not None:
        action.dataloader.sec_source.get_reporting_currency.side_effect = side_effect
    else:
        action.dataloader.sec_source.get_reporting_currency.return_value = currency_result
    return action


def test_returns_currency_from_sec_source():
    action = _fake_action(currency_result="USD")

    assert resolve_reporting_currency(action, "AAPL") == "USD"
    action.dataloader.sec_source.get_reporting_currency.assert_called_once_with("AAPL")


def test_returns_none_when_sec_source_returns_none():
    action = _fake_action(currency_result=None)

    assert resolve_reporting_currency(action, "SAP") is None


def test_swallows_exception_and_returns_none():
    action = _fake_action(side_effect=ConnectionError("SEC down"))

    result = resolve_reporting_currency(action, "AAPL")  # darf nicht werfen

    assert result is None
