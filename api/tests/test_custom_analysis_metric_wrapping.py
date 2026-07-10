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


def test_dict_shaped_metric_criterion_matches_displayed_field_by_key_substring():
    """`analyze_payout_ratio` (result_shape="dict") returns
    {"symbol", "payout_ratio", "threshold", "status", "message"} - before
    this fix, a criterion on this metric was silently never evaluated
    because `safe` (the whole dict) is never int/float. The frontend's
    pickPrimaryObjectValue (ComparePivotTable.tsx) already displays
    "payout_ratio" for this metric via a key-substring match; the backend
    must pick the same field for the badge to agree with what's shown."""
    raw = {"symbol": "AAPL", "payout_ratio": 42.0, "threshold": 75.0, "status": "ok", "message": "..."}

    result = _wrap_metric_result(raw, criterion={"operator": "<", "threshold": 75}, metric_key="analyze_payout_ratio")

    assert result["meets_criterion"] is True
    json.dumps(result)


def test_dict_shaped_metric_criterion_uses_tbv_and_price_special_case():
    raw = {"symbol": "AAPL", "tbv_per_share": 5.0, "current_price": 12.0, "price_to_tbv": 2.4}

    result = _wrap_metric_result(
        raw, criterion={"operator": ">", "threshold": 5}, metric_key="get_current_tbv_and_price"
    )

    assert result["meets_criterion"] is False  # 2.4 is not > 5
    json.dumps(result)


def test_dict_shaped_metric_criterion_falls_back_to_first_numeric_field():
    """`calculate_avg_quarterly_profit_growth` returns "avg_growth" first,
    with no key-substring match against the metric key - must fall back to
    the first numeric field (matches the frontend's numericMatch fallback)."""
    raw = {
        "avg_growth": 3.2,
        "symbol": "AAPL",
        "actual_start_date": "2024-03-31",
        "actual_end_date": "2025-03-31",
        "frequency": "quarterly",
        "message": "...",
    }

    result = _wrap_metric_result(
        raw, criterion={"operator": ">", "threshold": 2}, metric_key="calculate_avg_quarterly_profit_growth"
    )

    assert result["meets_criterion"] is True
    json.dumps(result)


def test_dict_shaped_metric_without_any_numeric_field_yields_no_badge():
    raw = {"symbol": "AAPL", "status": "ok", "message": "no numbers here"}

    result = _wrap_metric_result(raw, criterion={"operator": ">", "threshold": 5}, metric_key="some_metric")

    assert "meets_criterion" not in result


def test_infinite_debt_free_result_still_evaluates_against_threshold():
    """calculate_net_debt_to_ebitda/calculate_interest_coverage_ratio return
    a bare float("inf") for the debt-free edge case; make_json_safe turns
    that into the string "inf" before a criterion ever sees it - without the
    inf-string handling in _evaluate_criterion, a threshold on this metric
    would silently never show a badge for exactly the best possible case."""
    result = _wrap_metric_result(float("inf"), criterion={"operator": ">=", "threshold": 3})

    assert result["value"] == "inf"
    assert result["meets_criterion"] is True
    json.dumps(result)


def test_metric_without_criterion_gets_no_badge_at_all():
    result = _wrap_metric_result(8.46, criterion=None, metric_key="calculate_KGV")

    assert "meets_criterion" not in result
