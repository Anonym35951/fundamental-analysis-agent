"""Regressionstest für `_wrap_metric_result` (api/routes/custom_analysis.py):
ein bare `None`-Rückgabewert (z. B. von `get_max_historical_stock_data`, das
nach ausgeschöpften Retries oder bei legitim fehlenden Daten `None` liefert)
fiel bisher durch alle Formerkennungs-Zweige und ergab `{"value": None}` ohne
"error" und ohne "series". Das Frontend liest eine leere Series als "nicht
chart-fähig" und stuft die Metrik dadurch stillschweigend als leere
Tabellenzeile ein, statt sie als fehlgeschlagen zu kennzeichnen.
"""
import pandas as pd

from api.routes.custom_analysis import _wrap_metric_result


def test_none_result_is_surfaced_as_error_not_silent_value():
    result = _wrap_metric_result(None, criterion=None)

    assert result["value"] is None
    assert "error" in result
    assert "series" not in result


def test_dataframe_result_still_produces_series():
    df = pd.DataFrame({"Close": [100.0, 101.5]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))

    result = _wrap_metric_result(df, criterion=None)

    assert "error" not in result
    assert len(result["series"]) == 2
    assert result["series"][0]["value"] == 100.0


def test_scalar_result_still_works():
    result = _wrap_metric_result(42.0, criterion={"operator": ">", "threshold": 10})

    assert result["value"] == 42.0
    assert result["meets_criterion"] is True
