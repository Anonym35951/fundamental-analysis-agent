"""Regressionstest für `_wrap_metric_result` (api/routes/custom_analysis.py):
ein bare `None`-Rückgabewert (z. B. von `get_max_historical_stock_data`, das
nach ausgeschöpften Retries oder bei legitim fehlenden Daten `None` liefert)
fiel bisher durch alle Formerkennungs-Zweige und ergab `{"value": None}` ohne
"error" und ohne "series". Das Frontend liest eine leere Series als "nicht
chart-fähig" und stuft die Metrik dadurch stillschweigend als leere
Tabellenzeile ein, statt sie als fehlgeschlagen zu kennzeichnen.
"""
import json

import numpy as np
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


def test_numpy_float_criterion_result_is_native_python_bool_and_json_safe():
    """Live-Bug (2026-07-10): eine individuelle Analyse mit KGV + Schwellenwert
    (< 10) crashte den ganzen Custom-Analysis-Job mit
    `TypeError: Object of type bool is not JSON serializable` beim Schreiben
    des Job-Snapshots in die JSONB-Spalte. Ursache: `calculate_KGV` liefert
    intern oft ein `numpy.float64` (aus einer Pandas-Series-Berechnung) statt
    einem nativen Python-Float; der Vergleich `numpy.float64 > threshold` in
    `_evaluate_criterion` ergibt dadurch ein `numpy.bool_`/`numpy.bool`, keine
    Python-bool-Subklasse - Pythons `json`-Encoder lehnt das ab, obwohl
    `numpy.float64` selbst (als echte float-Subklasse) klaglos serialisiert.
    Compare setzt nie einen Schwellenwert, daher trat der Bug dort nie auf."""
    numpy_kgv = np.float64(8.5)  # < 10, numpy-typisiert wie ein echtes KGV-Ergebnis

    result = _wrap_metric_result(numpy_kgv, criterion={"operator": "<", "threshold": 10})

    assert result["meets_criterion"] is True
    assert type(result["meets_criterion"]) is bool  # nicht numpy.bool_/numpy.bool
    json.dumps(result)  # darf nicht mehr TypeError werfen


def test_numpy_bool_direct_result_is_sanitized_by_make_json_safe():
    from api.utils.json_sanitize import make_json_safe

    sanitized = make_json_safe(np.bool_(True))

    assert sanitized is True
    assert type(sanitized) is bool
    json.dumps(sanitized)
