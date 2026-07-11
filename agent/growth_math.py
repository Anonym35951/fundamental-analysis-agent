"""Reine Wachstumsraten-Mathematik, ohne Abhängigkeit auf Model/DataLoader
(vermeidet einen Zirkelimport zwischen den beiden - Model importiert bereits
DataLoader).

Produktentscheidung (2026-07-11, Block 5 der LAUNCH.md-Methodik-Runde):
AAGR/AQGR nutzten bisher das arithmetische Mittel der Einzelperioden-
Wachstumsraten - das verzerrt systematisch nach oben bei volatilen Gewinnen
(+100% dann -50% ergibt im Mittel +25%, obwohl der Wert am Ende unverändert
ist) und AQGR verglich zusätzlich Quartal-zu-Vorquartal ohne
Saisonbereinigung. compute_net_income_cagr ersetzt das durch eine einzige
CAGR-Berechnung aus dem ältesten und dem neuesten verfügbaren positiven
Datenpunkt, mit dem echten Zeit-Delta (nicht der Periodenanzahl) als
Exponent - das löst beide Probleme gleichzeitig: kein Mittelwert-Bias mehr,
und der Vergleich über den vollen Kalenderzeitraum ist von der
Quartalsposition unabhängig.
"""
from __future__ import annotations

import pandas as pd


def compute_net_income_cagr(net_incomes: pd.Series) -> dict:
    """Berechnet die annualisierte CAGR aus dem ältesten und dem neuesten
    verfügbaren positiven Datenpunkt einer Nettogewinn-Serie.

    Args:
        net_incomes: Series mit DatetimeIndex (beliebige Sortierrichtung),
            bereits NaN-bereinigt.

    Returns:
        Erfolg: {"cagr": float (Prozent), "years": float, "start_date":
                 Timestamp, "end_date": Timestamp, "start_value": float,
                 "end_value": float}
        Fehler: {"error": str}
    """
    s = net_incomes.sort_index(ascending=True)  # chronologisch, alt -> neu
    if len(s) < 2:
        return {"error": "Nicht genügend Datenpunkte für CAGR (mindestens 2 erforderlich)."}

    # Walk-inward von beiden Rändern: ein einzelner negativer Randwert (z. B.
    # ein Verlustquartal am Anfang oder Ende der Historie) soll die CAGR-
    # Berechnung nicht komplett verhindern, solange genug positive Werte
    # dazwischen liegen. Strukturell mehrheitlich negative Serien werden
    # bereits vor diesem Aufruf gesondert behandelt (siehe Model.py) - hier
    # geht es nur um einzelne Rand-Ausreißer.
    pos_idx = [i for i, v in enumerate(s.values) if v > 0]
    if len(pos_idx) < 2:
        return {"error": "Nicht genügend positive Datenpunkte für CAGR (mindestens 2 erforderlich)."}

    start_i, end_i = pos_idx[0], pos_idx[-1]
    if end_i <= start_i:
        return {"error": "Kein gültiges chronologisches Paar positiver Werte für CAGR."}

    start_date, end_date = s.index[start_i], s.index[end_i]
    start_value, end_value = float(s.iloc[start_i]), float(s.iloc[end_i])

    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return {"error": "Zeitspanne zwischen ältestem und neuestem Datenpunkt ist 0 oder negativ."}

    cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
    return {
        "cagr": cagr,
        "years": years,
        "start_date": start_date,
        "end_date": end_date,
        "start_value": start_value,
        "end_value": end_value,
    }
