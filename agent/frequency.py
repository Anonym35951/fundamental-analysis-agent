"""EVOLVING.md EV-130: einzige Quelle der Wahrheit für die im Projekt
erlaubten Perioden-Frequenzen.

Ersetzt die ~30 verstreuten Literale `["annual", "quarterly"]` in
agent/DataLoader.py, agent/Model.py, agent/data_sources/sec_source.py und
agent/AgentAction.py durch einen gemeinsamen Import. Dieser Schritt ist
bewusst rein mechanisch (verhaltensneutral, EVOLVING.md §2): jede
Fehlermeldung/jedes Fehler-Dict an den ~30 Guard-Stellen bleibt exakt
unverändert (siehe agent/tests/test_golden_master_frequencies.py, das genau
diese Texte fixiert) - nur die Werte-Menge, gegen die geprüft wird, kommt
jetzt aus `ALLOWED_FREQUENCIES` statt aus einem lokalen Listen-Literal.

EV-132 erweitert `ALLOWED_FREQUENCIES` um "ttm" (Delegation auf den
bestehenden Annual-Pfad - siehe agent/Model.py). Diese eine Zeile ist damit
auch der Kill-Switch: "ttm" aus dem Tupel entfernen deaktiviert das Feature
vollständig, ohne dass irgendein Guard-Aufruf angepasst werden müsste.

Wichtig (siehe agent/tests/test_golden_master_frequencies.py,
test_golden_master_invalid_frequency_error_text): nicht jede Model.py-Methode
hat überhaupt einen Guard auf dieser Werte-Menge - `calculate_kuv` etwa prüft
nur `if frequency == "annual"` und behandelt jeden anderen String (auch
zukünftig "ttm") als "quarterly". Das Hinzufügen von "ttm" zu
ALLOWED_FREQUENCIES allein macht `frequency="ttm"` daher noch NICHT für jede
Methode korrekt - EV-132 setzt stattdessen `resolve_ttm_alias()` einzeln in
jeder der ~15 Methoden aus TTM_CAPABLE_METRICS (unten) ein, VOR jedem Guard
und jeder Cache-Key-Berechnung.
"""
from __future__ import annotations

# Bewusst noch OHNE "ttm" - EV-132 erweitert dieses Tupel, erst nachdem die
# Capability-Map (unten) und die Delegation je Metrik stehen.
ALLOWED_FREQUENCIES: tuple[str, ...] = ("annual", "quarterly")

# Separates Tupel für die 5 "Whole-Mode"-Analysen in agent/AgentAction.py
# (analyze_wachstumswerte, analyze_typical_cyclers, analyze_cycler_turnarounds,
# analyze_optionality, analyze_asset_play) - bleibt bewusst zweiwertig
# (kein "ttm" hier), weil der explizite Guard in analyze_wachstumswerte
# gegen GENAU dieses Tupel prüft. Audit-Ergebnis (EV-135, siehe
# agent/tests/test_ttm_mode_methods_ev135.py):
#
# - analyze_wachstumswerte hat als einzige der 5 eine riskante interne
#   Verzweigung (`if frequency == "annual"` entscheidet AAGR vs. AQGR sowie
#   diverse Label-Texte). Deshalb übersetzt diese Methode "ttm" VOR ihrem
#   eigenen Guard via resolve_ttm_alias() auf "annual" - der Guard selbst
#   sieht "ttm" nie und muss darum nicht erweitert werden.
# - Die anderen 4 reichen `frequency` nur unverändert an bereits TTM-fähige
#   Model.py-Methoden durch (calculate_roe, calculate_cashflow_margin,
#   calculate_debt_to_equity, calculate_net_debt_to_ebitda,
#   calculate_inventory_to_revenue_ratio, calculate_ev_to_ebitda,
#   calculate_cash_to_market_cap, calculate_current_netCurrentAssets - alle
#   in TTM_CAPABLE_METRICS) und haben selbst KEINEN Guard gegen dieses
#   Tupel - "ttm" fließt dort bereits sicher durch, ohne Codeänderung.
#
# Dieses Tupel bleibt trotzdem als Dokumentations-Anker bestehen: falls
# künftig eine der 5 Methoden eine NEUE, noch nicht auditierte interne
# `if frequency ==`-Verzweigung bekommt, muss sie hier (und im Audit-Test)
# geprüft werden, bevor "ttm" für sie freigegeben wird.
MODE_ALLOWED_FREQUENCIES: tuple[str, ...] = ("annual", "quarterly")


def resolve_ttm_alias(frequency: str) -> str:
    """EV-132: 'ttm' delegiert 1:1 auf den bestehenden 'annual'-Pfad, der für
    die meisten aktuellen Kennzahlen bereits TTM (Summe der letzten vier
    Quartale) berechnet (siehe EVOLVING.md §4.3/§9 - Model.calculate_kuv,
    calculate_roe et al.). Jede andere Eingabe (u. a. 'quarterly' und
    ungültige Werte) kommt unverändert zurück - die aufrufende Methode
    validiert/behandelt sie exakt wie vor dieser Änderung.

    Aufrufkonvention in den ~15 TTM-fähigen Model.py-Methoden (siehe
    CAPABILITY_MAP unten): als allererste Zeile im Funktionskörper aufrufen,
    den ORIGINAL angeforderten Wert für das Antwort-Dict separat merken -
    sonst zeigt die Antwort fälschlich "annual" statt "ttm":

        requested_frequency = frequency
        frequency = resolve_ttm_alias(frequency)
        ...
        return {..., "frequency": requested_frequency}

    Dadurch sehen sämtliche nachgelagerten Guards, Cache-Keys
    (data_type = f"..._{frequency}") und Downstream-Aufrufe an
    DataLoader/SecSource ausschließlich "annual" - kein einziger der dortigen
    ~19 Guards oder Cache-Namespaces muss "ttm" kennen oder anfassen."""
    return "annual" if frequency == "ttm" else frequency


# EV-131: Capability-Map - welche vom Nutzer via metric_catalog auswählbaren
# Kennzahlen "ttm" akzeptieren (Delegation auf den Annual-Pfad, siehe
# resolve_ttm_alias oben) und welche nicht. Bewusst als Allowlist (nicht
# "alle außer den unten genannten"): eine neu hinzugefügte Kennzahl ohne
# Eintrag hier bekommt kein "ttm" angeboten, bis sie explizit geprüft wurde.
#
# Ausschlüsse laut TTM-Kennzahlenmatrix (EVOLVING.md §9), v1 bewusst kein
# TTM: Dividenden-Kennzahlen (calculate_current_dividend_yield hat gar
# keinen frequency-Param; calculate_historical_dividend_yield_average,
# analyze_payout_ratio, analyze_dividend_history ebenso), Ø Jahres-/
# Quartalswachstum (calculate_avg_annual_profit_growth,
# calculate_avg_quarterly_profit_growth - TTM-vs-TTM-Vorjahr wäre neue
# Numerik, kein reiner Alias) sowie alles ohne frequency-Parameter im
# Katalog (KGV, PEG, Buchwert/Aktie, TBV&Kurs, Inflation, Bandbreiten/CRV/
# Kursziele - diese bauen auf den bereits-TTM-historischen Reihen auf und
# haben keine eigene frequency-Auswahl).
TTM_CAPABLE_METRICS: frozenset[str] = frozenset({
    "calculate_kuv",
    "calculate_ev_to_sales",
    "calculate_ev_to_ebit",
    "calculate_ev_to_ebitda",
    "calculate_price_to_ebit",
    "calculate_price_to_freeCashflow",
    "calculate_roe",
    "calculate_ROIC",
    "calculate_cashflow_margin",
    "calculate_debt_to_equity",
    "calculate_net_debt_to_ebitda",
    "calculate_interest_coverage_ratio",
    "calculate_inventory_to_revenue_ratio",
    "calculate_current_netCurrentAssets",
    "calculate_cash_to_market_cap",
})
