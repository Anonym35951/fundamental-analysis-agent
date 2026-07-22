# Changelog

Alle nennenswerten Änderungen an ComAnalysis werden hier festgehalten.

## [1.1.0] - 2026-07-22

### Geändert
- **Historische Marktkapitalisierung deutlich verdichtet**: Der Chart zeigte bei kleinen Zeiträumen
  (1J/2J/5J) bisher nur die wenigen quartalsweisen SEC-Berichtsstichtage (1J oft nur 2 Punkte, quasi
  eine Gerade zwischen zwei Werten). Jetzt basiert die Berechnung auf täglichen Kursen
  (AlphaVantage `TIME_SERIES_DAILY_ADJUSTED`, mit Yahoo-Tagesdaten als Ausfall-Absicherung) statt nur
  quartalsweisen Kurspunkten, altersgestuft verdichtet (täglich innerhalb der letzten 2 Jahre,
  wöchentlich bis 5 Jahre, monatlich darüber). „Max" bleibt dabei so kompakt wie zuvor, 1J/2J/5J zeigen
  jetzt den tatsächlichen Verlauf innerhalb des Zeitraums statt nur Start- und Endpunkt.
- Im Zuge dessen eine dadurch aufgedeckte Regression beim historischen Enterprise Value behoben (wäre
  mit der neuen, nicht mehr rein quartalsweisen Marktkap-Serie sonst leer geblieben) — Enterprise Value
  bleibt weiterhin quartalsweise dargestellt.

## [1.0.1] - 2026-07-22

### Hinzugefügt
- **Umschaltbare Chart-Darstellung (Linie ↔ Säulen)** für fachlich geeignete Charts (Flow-Kennzahlen:
  Umsatz, EBIT, EBITDA, Nettoergebnis, Free Cashflow, operativer Cashflow):
  - Dezentes Dropdown ("Darstellung: Linie/Säulen") oben rechts an jedem geeigneten Chart, a11y-vollständig
    (Tastatur, Screenreader, `aria-*`).
  - AnalysePage (Einzelfirma) und ComparePage (Firmenvergleich, gruppierte Säulen bis 4 Firmen gleichzeitig,
    darüber automatisch Linie) unterstützt.
  - Reine Visualisierungsschicht: identische Daten, Berechnung, Zeitraum, Währung und %-Veränderung in
    beiden Darstellungen; kein zusätzlicher API-Request beim Wechsel; korrekte Nulllinie bei negativen
    Werten (z. B. Verlustjahre); Default bleibt Linie (bestehendes Verhalten unverändert).
  - Mobile-optimiert (responsive Balkenbreite/-abstand je Viewport und Firmenanzahl).

### Geändert
- Deckkraft des Chart-Tooltips erhöht (nahezu deckender statt halbtransparenter Hintergrund) für bessere
  Lesbarkeit der angezeigten Firmenwerte vor bewegten Chart-Linien/-Balken.

Details und Implementierungs-Nachweise siehe `EVOLVING.md`, Abschnitt "Erweiterbare Chart-Darstellungen –
Linie & Säulen".
