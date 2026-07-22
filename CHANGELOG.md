# Changelog

Alle nennenswerten Änderungen an ComAnalysis werden hier festgehalten.

## [1.0.01] - 2026-07-22

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
