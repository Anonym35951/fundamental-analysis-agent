"""Regressionstest für EVOLVING.md EV-001/EV-030: Datums-Serialisierung in
`_wrap_metric_result` (api/routes/custom_analysis.py).

Root Cause des Tooltip-Bugs (EVOLVING.md Abschnitt 5.3/6, P5), URSPRÜNGLICH
bestätigt durch diesen Test: `str(idx)` auf einem pandas-DatetimeIndex ergab
den VOLLEN Timestamp-String ("2024-09-30 00:00:00"), nicht nur das Datum.
EV-030 hat das auf `idx.strftime("%Y-%m-%d")` umgestellt (sauberes Datum,
kein Zeitanteil mehr) - dieser Test prüft jetzt das NEUE Verhalten.

Wichtig: Das saubere Datumsformat allein löst die Merge-Kollision noch
NICHT vollständig - zwei Firmen mit unterschiedlichen Fiskal-Stichtagen im
selben Kalenderjahr (z. B. 30.09. vs. 31.12.) erzeugen weiterhin disjunkte
exakte Datums-Schlüssel. Das eigentliche Zusammenführen passiert erst durch
das Perioden-Bucketing im Frontend (chartUtils.ts::bucketKey/mergeLayers,
ebenfalls EV-030) - dieser Test hält das bewusst fest, damit klar bleibt,
welcher Teil wo behoben wird.

Ergänzt test_custom_analysis_metric_wrapping.py (dort wird nur Länge/Wert
der Series geprüft, nicht das Datumsformat selbst)."""
import pandas as pd

from api.routes.custom_analysis import _wrap_metric_result


def test_dataframe_series_date_is_clean_yyyy_mm_dd_post_ev030():
    df = pd.DataFrame(
        {"Close": [100.0, 101.5]},
        index=pd.to_datetime(["2024-09-30", "2024-12-31"]),
    )

    result = _wrap_metric_result(df, criterion=None)

    dates = [point["date"] for point in result["series"]]
    # Nachher (EV-030): reines Datum, kein Mitternachts-Zeitanteil mehr.
    assert dates == ["2024-09-30", "2024-12-31"]


def test_non_datetime_index_falls_back_to_str():
    """Seltener, aber möglicher Fall: ein nicht-datetime-artiger Index
    (z. B. eine reine Integer-Periodenzahl) hat kein .strftime - str(idx)
    bleibt als Fallback erhalten, damit dieser Pfad nicht bricht."""
    df = pd.DataFrame({"Close": [100.0]}, index=[2024])

    result = _wrap_metric_result(df, criterion=None)

    assert result["series"][0]["date"] == "2024"


def test_two_companies_with_different_fiscal_dates_still_yield_disjoint_exact_date_keys():
    """Demonstriert, was EV-030s Backend-Änderung ALLEIN nicht löst: zwei
    Firmen mit versetzten Fiskal-Stichtagen im selben Kalenderjahr erzeugen
    weiterhin disjunkte exakte Datums-Schlüssel ("2024-09-30" vs.
    "2024-12-31") - das eigentliche Zusammenführen in eine gemeinsame Zeile
    passiert erst durch das Perioden-Bucketing im Frontend
    (chartUtils.ts::bucketKey/mergeLayers), nicht durch das reine
    Datumsformat."""
    company_a = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2024-09-30"]))
    company_b = pd.DataFrame({"Close": [200.0]}, index=pd.to_datetime(["2024-12-31"]))

    dates_a = [p["date"] for p in _wrap_metric_result(company_a, criterion=None)["series"]]
    dates_b = [p["date"] for p in _wrap_metric_result(company_b, criterion=None)["series"]]

    assert set(dates_a).isdisjoint(set(dates_b))


def test_first_non_index_column_is_used_as_value_when_multiple_columns_present():
    df = pd.DataFrame(
        {"Close": [100.0], "Volume": [123456]},
        index=pd.to_datetime(["2024-09-30"]),
    )

    result = _wrap_metric_result(df, criterion=None)

    assert result["series"][0]["value"] == 100.0


def test_dataframe_without_columns_yields_empty_series():
    df = pd.DataFrame(index=pd.to_datetime(["2024-09-30"]))

    result = _wrap_metric_result(df, criterion=None)

    assert result["series"] == []
