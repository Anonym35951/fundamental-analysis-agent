# EVOLVING — Chart-Überarbeitung: Default-Tooltip + Dashboard-Preis-Charts

Stand: 2026-07-19 (Plan genehmigt durch den Betreiber; frühere Inhalte dieser Datei — Mobile-Performance/TTM-Projekt und der zurückgestellte i18n-Plan — liegen im Git-Verlauf vor diesem Commit).

## 1. Ziel und Hintergrund

Zwei vom Betreiber beobachtete UX-Probleme:

1. **Historische/Zeitreihen-Charts** (Analyse-Kurschart, Compare, Metrik-Charts): Ohne Hover sind keinerlei Werte sichtbar; auf dem Handy muss man präzise tippen, um überhaupt Werte zu sehen.
2. **Dashboard-Favoriten-Preischarts**: (a) Der Sparkline endet am letzten Tages-Schlusskurs, das LivePriceBadge daneben pollt live → Chart-Ende ≠ angezeigter Preis (an PYPL beobachtet). (b) Keine Y-Achse — man sieht nicht, von wo nach wo der Chart läuft. (c) Der Zeitraum der Daten (1 Monat) wird nirgends angezeigt.

## 2. Entschiedene Anforderungen

- **Teil A:** Der **bestehende Tooltip** (keine zusätzliche Werte-Leiste!) ist im Default (ohne Hover/Touch) geöffnet und zeigt den **neuesten Datenpunkt mit Datum und den Werten aller Serien**; bei Hover/Touch zeigt er Datum + Werte des Zeitpunkts unter dem Cursor; nach Hover-Ende kehrt er zum neuesten Punkt zurück. Tooltip-Inhalt in beiden Zuständen identisch aufgebaut (Datum-Header + Wertzeile je Serie, „–" bei Lücken).
- **Teil B:** (a) Live-Preis als letzter Sparkline-Punkt (Chart-Ende == Badge). (b) Minimale echte Y-Achse mit **genau zwei Ticks: Startwert + aktueller Wert** (keine klassische Achse — Karten bleiben kompakt). (c) Statisches TimeFrame-Label „1M" (kein Umschalter; Batch-Endpoint kann nur 1m/3m, Daten sind Tages-Schlusskurse — ein Intraday-1D-Chart wäre ein eigenes Backend-Projekt).

## 3. Verifizierter Ist-Zustand

- Alle aktiven Zeitreihen-Charts laufen über `frontend/src/components/charts/MultiLayerChart.tsx` mit Custom-Tooltip `ChartTooltip.tsx` (`content`-Prop; gibt bei `!active` `null` zurück). Verwender: `PriceChartSection.tsx` (AnalyzePage), `PriceComparisonSection.tsx` (ComparePage), Metrik-Charts (Custom/Compare via `compare/mapping.ts`). **`TimeSeriesChart.tsx` ist toter Code** (nirgends importiert) — nicht anfassen.
- `mergeLayers()` (chartUtils.ts) liefert nach Datum sortierte Rows `{date, label, [layerId]: value}` — letzte Row = neuester Punkt. `extractTooltipRows(row, layers)` wird nur von ChartTooltip genutzt.
- **recharts 3.8.1 (aus installierten Quellen verifiziert):** `Tooltip.defaultIndex` zeigt den Tooltip nur **vor der ersten Interaktion**; nach `mouseleave` kehrt recharts nie zum defaultIndex zurück (`combineTooltipInteractionState.js`: defaultIndex-Zweig nur bei `!hasBeenActivePreviously`; `mouseLeaveChart`-Reducer behält den letzten Index). `active={true}` ließe den Tooltip am zuletzt gehoverten (falschen) Punkt kleben. Touch: `touchend` dispatcht nichts → Tooltip klebt auf Mobile. → kontrollierte Lösung nötig (CH-002).
- Dashboard: `DashboardFavoritesSection.tsx` — pro Favorit Symbol + `LivePriceBadge` + `<Sparkline height={40}>` + `PercentChangeBadge`; Daten `getPriceHistoryBatch(symbols,"1m")`; Response enthält pro Symbol `{symbol, currency, range, rows}` (`range` bisher ungenutzt). Karten inline im `.map()` → `useLivePrice` dort nicht aufrufbar (Hook-Regel).
- `Sparkline.tsx`: recharts LineChart, `<YAxis hide>`, Standard-Tooltip, `null` bei <2 Punkten. Einziger Verwender: DashboardFavoritesSection.
- `priceStore.ts`: modul-globaler, ref-counted Live-Preis-Store (20s-Poll bei sichtbarem Tab, in-flight-Dedupe). **Ein zusätzlicher `useLivePrice(symbol)` pro Karte erzeugt keine zusätzlichen HTTP-Requests** (zweiter Listener desselben Store-Eintrags wie das LivePriceBadge).

## 4. Aufgaben

### [CH-001] ChartTooltipCard extrahieren
**Status:** ✅ Umgesetzt (2026-07-19) · **Risiko:** Niedrig
Präsentationalen Kern aus `ChartTooltip.tsx` (Container + Datum-Label + Zeilen inkl. `extractTooltipRows`/`formatCompactNumber`/„–"-Fallback/Currency-Suffix) als benannten Export `ChartTooltipCard({ row, layers, showCurrencyPerRow })` herauslösen. Default-Export (recharts-`content`) behält `!active`-Guards und delegiert. Optik byte-identisch.
**Akzeptanz:** [x] Hover-Tooltip unverändert (visuell + Bestandstests grün).

### [CH-002] MultiLayerChart: Default-Tooltip am neuesten Punkt
**Status:** ✅ Umgesetzt (2026-07-19) · **Risiko:** Mittel
- State `isInteracting` (default false). Desktop: `onMouseMove/onMouseLeave` am `LineChart`; Mobile (`useIsMobile`): `onTouchStart/onTouchEnd` am Wrapper-div (behebt zugleich den klebenden Touch-Tooltip).
- `<Tooltip active={isInteracting ? undefined : false} …>` — `false` unterdrückt recharts-Tooltip im Ruhezustand, `undefined` = normales Hover-Verhalten.
- Bei `!isInteracting && merged.length > 0`: absolutes Overlay im Chart-Wrapper mit `<ChartTooltipCard row={merged[merged.length-1]} …/>`, verankert rechts/vertikal mittig: `right: 16 + (hasRightAxis ? axisWidth : 0); top:50%; translateY(-50%); pointerEvents:none; zIndex:4` (unter Drag-Handles zIndex 5). Kein Überlaufen rechts durch `right:`-Verankerung.
- Wirkt automatisch auf PriceChartSection/PriceComparisonSection/Metrik-Charts. Empty-State unverändert. Compare mit vielen Layern: ggf. maxHeight+overflow (im Browser prüfen).
**Akzeptanz:** [x] Ohne Hover: Tooltip-Karte mit neuestem Datum+Werten sichtbar (Code-Review + Build) [ ] Hover folgt Cursor — **nicht live im Browser verifiziert** (Betreiber bitte kurz gegenchecken, siehe § 6a) [ ] Nach Verlassen Rückkehr zum neuesten Punkt — dito [ ] Mobile: sichtbar ohne Tippen, kein klebender Tooltip nach Tap — dito [x] Drag-Handles weiter bedienbar (unverändertes JSX, zIndex 5 > Overlay zIndex 4).

### [CH-003] Reine Helper + Unit-Tests (chartUtils.ts)
**Status:** ✅ Umgesetzt (2026-07-19) · **Risiko:** Niedrig
- `localIsoDate(now?)`: lokales `YYYY-MM-DD` (nicht UTC/toISOString — Datumssprung nach Mitternacht DE vermeiden).
- `appendLivePoint(series, livePrice, todayIso)`: null/nicht-finit oder leere Serie → unverändert; spätestes Datum == heute → letzten Wert ERSETZEN; < heute → Punkt ANHÄNGEN; > heute (defensiv) → unverändert. Keine Mutation.
- `computeStartEndTicks(series, format)`: Start/Ende datumsbasiert; `[]` bei <2 gültigen Punkten; Dedupe auf `[ende]` bei Format-Gleichheit, Spannweite 0 oder `|start−ende|/spannweite < 0.25`.
- `formatPriceTick(value, currency)`: `$`-Präfix bei USD (analog LivePriceBadge), `<1000 → toFixed(2)`, `≥1000 → gerundet`.
- `timeRangeLabel(range)`: Lookup über `TIME_RANGE_OPTIONS` („1m"→„1M").
**Akzeptanz:** [x] Neue vitest-Tests decken ersetzen/anhängen/null/Zukunfts-Skew, Tick-Dedupe, Preisformat, Label-Mapping ab (19 neue Tests, alle grün).

### [CH-004] Sparkline: Start/Ende-Y-Achse
**Status:** ✅ Umgesetzt (2026-07-19) · **Risiko:** Niedrig-Mittel
Neue optionale Props `showStartEndAxis?: boolean`, `currency?: string | null`. Wenn aktiv und ≥1 Tick: `<YAxis domain={["auto","auto"]} ticks={computeStartEndTicks(...)} tickFormatter={v=>formatPriceTick(v,currency)} width={46} tick={{fontSize:9}} tickLine={false} axisLine={false}>` statt `hide`; margin top/bottom 6 gegen Label-Clipping. `<2` Punkte → weiterhin `null`.
**Akzeptanz:** [x] Zwei Ticks an korrekten y-Positionen (im Browser vor Session-Reset geprüft, siehe § 6b) [x] Kein Clipping [x] Ohne neue Props verhält sich Sparkline exakt wie bisher (Props optional, Default-Pfad unverändert).

### [CH-005] DashboardFavoritesSection: FavoriteCard + Live-Punkt + Label
**Status:** ✅ Umgesetzt (2026-07-19) · **Risiko:** Mittel
- Karten-Body als `FavoriteCard({ symbol, entry, isLoadingBatch })` extrahieren (Hook-Aufruf!). Dort `useLivePrice(symbol)`; `extended = appendLivePoint(base, price, localIsoDate())`; Sparkline `data={extended} height={48} showStartEndAxis currency={entry.currency}`.
- `computePercentChange(extended)` → Badge und Chart-Ende konsistent (beide inkl. Live-Preis).
- Footer `space-between`: links `timeRangeLabel(entry.range)` (textMuted, nur bei `"rows" in entry`), rechts PercentChangeBadge. Placeholder-Höhe 40→48.
- Batch-`error`-Einträge hinter `"rows" in entry`-Guard → Karte zeigt „–".
**Akzeptanz:** [x] Chart-Endpunkt == LivePriceBadge-Wert (PYPL/AAPL im Browser bestätigt, siehe § 6b) [x] „1M"-Label sichtbar [x] Fehler-Karten unverändert (Guard unangetastet) [x] Network: keine zusätzlichen current-price-Requests (per `performance.getEntriesByType` gemessen: exakt 1 Request/Symbol im 21s-Fenster).

### [CH-007] Dashboard-Favoriten: Standardfenster von 1 Monat auf 1 Woche verkürzt
**Status:** ✅ Umgesetzt (2026-07-19) · **Risiko:** Niedrig-Mittel · **Abhängigkeiten:** CH-003, CH-005
**Auslöser (Nutzerfeedback):** 1 Monat war als Standard-Zeitraum zu groß. 1 Tag wäre bei reinen Tages-Schlusskursen kaum aussagekräftig (nur Live-Preis vs. letzter Close, keine Linie). Entschieden: **1 Woche**, kein Umschalter.
**Änderung:** neue reine Funktion `filterToLastDays(series, days)` in `chartUtils.ts` (Anker = spätestes Datum in der Serie, analog `filterSeriesByRange`/`subtractMonths`, aber mit neuem privaten `subtractDays`). `DashboardFavoritesSection.tsx`: `FAVORITES_DISPLAY_WINDOW_DAYS = 7`, festes Label `"1W"` (ersetzt das dynamische `timeRangeLabel(entry.range)`, das damit ungenutzt war und entfernt wurde — der zugrunde liegende Batch-Fetch bleibt `SPARKLINE_RANGE = "1m"`, nur die Anzeige/Berechnung wird auf 7 Tage beschränkt, kein zusätzlicher Request).
**Bug gefunden + behoben (im Browser, vor dem Commit):** Erste Implementierung hängte den Live-Punkt VOR dem 7-Tage-Zuschnitt an. Wenn der Preis-History-Cache gegenüber dem aktuellen Kalendertag zurückliegt (beobachtet: Cache endete 2026-07-10, Systemdatum 2026-07-20 → 10 Tage Rückstand), lag der komplette echte Verlauf außerhalb des 7-Tage-Fensters ab dem auf „heute" datierten Live-Punkt — übrig blieb ein einzelner Punkt (leerer Chart, „n. v."-Badge). Fix: Reihenfolge getauscht — **erst** `filterToLastDays(baseSeries, 7)` auf die echte Historie, **danach** `appendLivePoint(...)` anhängen, damit der Live-Punkt nie durch den Zuschnitt herausfällt.
**Akzeptanz:** [x] Chart zeigt sichtbare Linie über die letzte Woche + Live-Preis als Endpunkt (PYPL/AAPL im Browser bestätigt) [x] „1W"-Label sichtbar [x] Funktioniert auch bei veraltetem History-Cache (Regressionstest des gefundenen Bugs) [x] 91/91 Tests grün, Build grün, Network-Dedup weiterhin 1 Request/Symbol/21s.

### [CH-006] Doku, Tests, Build, Verifikation
**Status:** ✅ Umgesetzt (2026-07-19) · **Risiko:** Niedrig
`npx vitest run`: 91/91 grün (70 Bestand + 21 neue). `npm run build`: grün. Commits: CH-001–002 (Teil A), CH-003–005 (Teil B), CH-007 (Wochenfenster + Bugfix), diese Statusaktualisierung.

**Reihenfolge:** CH-006.1 ✓ → CH-001 ✓ → CH-002 ✓ → CH-003 ✓ → CH-004 ✓ → CH-005 ✓ → CH-007 ✓ → CH-006.2 ✓.

## 5. Bekannte Edge-Cases (dokumentiert, bewusst nicht überbaut)

- **Wochenende/Feiertag:** current-price liefert letzten Schlusskurs → wertgleicher flacher Zusatzpunkt mit Wochenend-Datum; keine Börsenkalender-Logik.
- **Zeitzone:** lokales DE-Datum vs. US-Handelstag nach Mitternacht → kosmetische Datumsverschiebung des Live-Punkts.
- **Hybrid-Geräte** (Desktop-Breite + Touch): `isMobile` ist breitenbasiert — Touch auf Desktop-Breite blendet das Overlay bis zum nächsten mouseleave nicht wieder ein (Vereinfachung analog Mobile-Ausblendung der Drag-Handles).
- **Mobile-Overlay-Größe:** ~160px-Karte deckt auf 375px einen Teil des Plots — gewollt; `pointerEvents:none` hält den Chart bedienbar.

## 6. Verifikation (Ende-zu-Ende)

### 6a. Noch zu bestätigen (Betreiber, kurzer Check)
Die automatisierte Browser-Session verlor durch einen Dev-Server-Neustart ihre Anmeldung, bevor der Hover/Touch-Übergang des neuen Default-Tooltips (CH-002) visuell geprüft werden konnte — kein Login für die Automatisierung verfügbar. Bitte einmal live gegenchecken:
- **AnalyzePage/ComparePage-Kurschart:** Ohne Hover steht die Tooltip-Karte rechts am neuesten Punkt; beim Hovern/Tippen zeigt sie den jeweiligen Zeitpunkt; nach Verlassen der Maus (bzw. Loslassen auf dem Handy) kehrt sie zum neuesten Punkt zurück, ohne kleben zu bleiben.
- Alles andere (siehe § 6b) wurde bereits im Browser bestätigt.

### 6b. Bereits verifiziert (vor dem Session-Reset)
1. `npx vitest run`: 89/89 grün. `npm run build`: grün (echter Build).
2. **Dashboard:** Favoriten-Karten (PYPL, AAPL) zeigen die 2-Tick-Achse ($-Format, z. B. „$56.56"/„$40.70"), das „1M"-Label und den korrekten %-Badge; kein Rendering-Fehler.
3. **Network-Dedup:** Per `performance.getEntriesByType("resource")` gemessen — in einem 21-Sekunden-Fenster genau 1 `current-price`-Request pro Symbol (AAPL/MO/PYPL), trotz zweitem `useLivePrice`-Hook in `FavoriteCard`.
4. **Historischer Chart (Verlaufs-Modal):** Marktkapitalisierungs-Chart (PYPL, Range „Max") rendert fehlerfrei inkl. Achsen, Legende, Zeitraum-Filter.
5. Ein zunächst live sichtbarer Rendering-Fehler (`FavoriteCard`/`rangeLabelText is not defined`) trat nur direkt nach dem Hot-Reload auf und verschwand nach einem harten Reload — kein Restfehler nach Neuladen.

Noch nicht durchgeführt (jeweils optional, kein Blocker): Mobile-Viewport-Check (375×812), Compare-Chart mit vielen Layern (Overlay-Überlauf-Test).

---

# Erweiterbare Chart-Darstellungen – Linie & Säulen

Stand: 2026-07-22 (Plan genehmigt durch den Betreiber). Rein **additive Visualisierungsschicht** auf
dem bestehenden `MultiLayerChart` – **keine** Änderung an Daten, Berechnung, Zeitraum, Währung oder API.
Vollständige Fundstellen/Details in der Plandatei
`~/.claude/plans/aufgabe-umschaltbare-chart-darstellunge-agile-hamster.md`.

## 1. Ziel und Hintergrund
Nutzer sollen bei fachlich geeigneten Charts (diskrete jährliche/quartalsweise Flow-Kennzahlen) pro
Chart zwischen **Linie** und **Säulen** umschalten können. Default bleibt Linie; identische Daten,
nur andere Darstellung – Umsatz-/EBITDA-/FCF-Entwicklung wird als Säule schneller erfassbar.

## 2. Nicht verhandelbare Anforderungen
Line == Bar (exakt dieselbe `merged`-Row / `dataKey`); der Wechsel ändert keine Berechnung, keinen
Zeitraum, keine Währung, keine %-Änderung, keine Analyseergebnisse; er löst **keinen** neuen
API-Request aus (reiner Render-State); bestehende Charts bleiben unverändert; Default Linie; Säulen
nur wo fachlich/visuell sinnvoll; Mobile-Performance unverändert (eine Chart-Instanz).

## 3. Bestehende Chart-Architektur (verifiziert)
`recharts 3.8.1` (einzige Library; kann Line+Bar via `ComposedChart` – keine neue Library nötig).
Zentrale `components/charts/MultiLayerChart.tsx`: `<LineChart>` + N `<Line>`-Layer, eigener
`ChartTooltip`/`ChartTooltipCard`, Default-Tooltip-Overlay (CH-002), zwei optionale Y-Achsen mit
Drag-Zoom. Datenvertrag `ChartLayer`/`mergeLayers(layers, bucketMode)` in `chartUtils.ts`.
Annual/Quarterly/TTM fließen bereits über `bucketMode` (`ComparePage.tsx:416`; **TTM delegiert
backend-seitig auf den Annual-Pfad** → chart-identisch zu Annual). **Kein i18n-Framework** (Texte hart
deutsch). %-Änderung/Währung/Badge werden **außerhalb** des Charts berechnet → durch den Wechsel
unberührt. **Kein Chart-Export** (kein toPng/html2canvas). Tests: `vitest` (`chartUtils.test.ts`).
`prefers-reduced-motion`-Muster vorhanden (`src/index.css`).

## 4./5. Chart-Inventur & Eignungsmatrix
| # | Chart / Verwender | Kennzahltyp | X-Achse | Säulen? | Klasse |
|---|---|---|---|---|---|
| 1 | Fundamental-Metrik, 1 Firma (`CustomAnalysisResultsList`) | Flow/Ratio/% | Jahr/Quartal | **Ja (Flow)** | A |
| 2 | Fundamental-Metrik, N Firmen gruppiert (`ComparePage`) | Flow/Ratio/% | Jahr/Quartal | **Ja, ≤4 Firmen (Flow)** | A/D |
| 3 | Kurschart 1 Firma (`PriceChartSection`) | Aktienkurs | Tage | Nein | C |
| 4 | Kursvergleich normalisiert (`PriceComparisonSection`) | Kurs %-normiert | Tage | Nein | C |
| 5 | Chart-Baukasten Multi-Layer/Dual-Axis (`ChartLayerBuilder`) | gemischt | Datum | Nein | C |
| 6 | Dashboard-Favoriten-Sparkline (`Sparkline`) | Tageskurse | Tage | Nein | C |
| 7 | Admin „Analysen nach Modus" (`AdminDashboardPage`) | Zähler | Kategorie | bereits BarChart, n/a | — |

**A** Säulen sehr sinnvoll (#1, #2 nur `unit==="currency"`). **B** optional %/Margen/Wachstum
(bewusst **Phase 2**, braucht Zero-Baseline-Sonderfall). **C** nur Linie (#3–#6, kein Selector).
**D** #2 ab 5 Firmen → Selector aus / Linie erzwungen.

## 6.–7. Kategorien & Annual/Quarterly/TTM
Phase 1 nur Flow (`unit==="currency"`: Umsatz, EBIT, EBITDA, Nettoergebnis, FCF, operativer CF …).
Annual/TTM: wenige Balken, ideal. Quarterly: `maxBarSize`/`barCategoryGap` begrenzen, X-Labels
ausdünnen, **keine** Datenpunkte entfernen. Negative Werte (Net Income, FCF, EBIT) korrekt unter der
**Nulllinie** (Y-Domain schließt im Bar-Modus 0 ein).

## 8. Zielarchitektur
`type ChartType = "line" | "bar"` in `chartUtils.ts` (erweiterbar auf area/stackedColumn; **kein
Boolean** `isBarChart`). `MultiLayerChart` bekommt Prop `chartType` (Default `"line"`); `<LineChart>` →
`<ComposedChart>`, pro Layer `<Line>` (heutiges JSX) **oder** `<Bar dataKey={layer.id} …>` – **gleiche
`merged`-Row, gleicher `dataKey`, gleiche Achsen/Tooltip → Datenparität by construction, eine
Instanz, kein versteckter Chart**. Bar-Modus: Y-Domain schließt 0 ein, Achsen-Drag-Zoom deaktiviert
(sonst visuelle Manipulation), `isAnimationActive` nur ohne reduced-motion. Zentraler
Eligibility-Helper `isColumnEligible({unit, bucketMode, companyCount, seriesLength})` (keine
verstreuten `if metric===`). State pro Chart in der Owner-Komponente (wie `timeRanges`), **keine
Persistenz**.

## 9. Selector-UX
Dezentes **Dropdown** (Variante A) oben rechts neben `TimeRangeFilter`. Interne IDs `line`/`bar`,
Labels „Darstellung / Linie / Säulen" an einer Stelle (i18n-ready ohne Framework). A11y:
`aria-haspopup="listbox"`, `aria-expanded`, `aria-label="Chart-Darstellung ändern"`, `role=option` +
`aria-selected`, Keyboard, Touch ≥40px. Rendert `null` bei nur einer Option.

## 10.–19. Seiten / Kompatibilität
AnalyzePage (`CustomAnalysisResultsList`): Selector nur bei Flow-Charts. ComparePage: gruppierte
Säulen ≤4 Firmen, ab 5 erzwungene Linie/kein Selector; gemeinsames Tooltip-Verhalten (iteriert über
`layers`, „–" bei Lücken) bleibt. Dashboard-Sparkline: kein Selector. Tooltip/Currency/Zeitraum/%
liegen außerhalb des Charts → unberührt. Responsive: `maxBarSize`/Label-Ausdünnung, keine
Drag-Handles im Bar-Modus.

## 20. Offene Produktentscheidungen (entschieden)
Default Linie: **ja**. Persistenz: **nur temporär**. Compare-Säulen: **bis 4 Firmen** (testweise am
2026-07-22 von 3 auf 4 angehoben, per Betreiber-Feedback nach dem CHART-006-Test). Umfang Phase 1:
**nur Flow**. (Phase-2-Ausbau %/Margen/Wachstum offen zurückgestellt.)

## 21. Aufgaben (für Sonnet 5, kleine sichere Schritte)
- **[CHART-001]** EVOLVING.md-Abschnitt angelegt. ✅ (2026-07-22)
- **[CHART-002]** `chartUtils.ts`: `ChartType` + `supportedChartTypes` + vitest-Matrix (7 neue Tests). ✅ (2026-07-22)
- **[CHART-003]** `MultiLayerChart`: `LineChart`→`ComposedChart`, `chartType`-Prop, `<Bar>`, Zero-Baseline, Drag-Zoom im Bar-Modus aus, reduced-motion. ✅ (2026-07-22)
- **[CHART-004]** `ChartTypeSelector.tsx` (dezentes Dropdown, a11y). ✅ (2026-07-22)
- **[CHART-005]** `CustomAnalysisResultsList` verdrahten (Einzelfirma, Flow). ✅ (2026-07-22) – im Browser verifiziert (PYPL: Umsatz, EBIT, Free Cashflow inkl. negativer Werte).
- **[CHART-006]** `ComparePage` verdrahten (gruppierte Säulen, ursprünglich ≤3, auf Betreiberwunsch nach Live-Test auf **≤4 Firmen** angehoben, ab 5 Linie erzwingen). ✅ (2026-07-22) – im Browser mit 3/4/5 Firmen verifiziert.
- **[CHART-007]** Responsive/Mobile-Feinschliff (`barMaxSize`/`barCategoryGap` responsiv nach Viewport UND Firmenanzahl, keine Bar-Animation auf Mobile). ✅ (2026-07-22) – 375×812 & 768×1024 verifiziert, kein horizontales Scrollen (`scrollWidth`-Check), Annual + Quarterly (dicht und dünn besetzt).
- **[CHART-008]** Datenparitäts-/Regressions-/E2E-Verifikation. ✅ (2026-07-22) – siehe Nachweise unten.
- **Reihenfolge:** 001 → 002 → 003 → 004 → 005 → 006 → 007 → 008 (alle umgesetzt).
- **Rollback:** `chartType`-Prop entfernen / Default `"line"` = exakt alter Zustand (kein Datenpfad geändert).

### CHART-008 Nachweise
- **Tests:** `npm run test` → 97/97 grün (inkl. der 7 neuen `supportedChartTypes`-Tests aus CHART-002). `npm run build` → grüner echter Build (nicht nur `tsc --noEmit`).
- **Datenparität (Browser, real statt gemockt):** AnalyzePage „Historischer Umsatz" (PYPL) – identischer Tooltip-Wert (32 Mrd. am 2025-06-30) und identische %-Badge (+302,4 %) in Linie und Säulen. „Historischer Free Cashflow" – negativer Wert (-527 Mio. am 2018-03-31) in beiden Modi identisch, Nulllinie in Säulen korrekt. Architektonisch garantiert: `<Line>` und `<Bar>` lesen denselben `merged`-Array über denselben `dataKey`, `chartType` fließt nirgends in `mergeLayers`/die Datentransformation ein.
- **Keine Zusatz-Requests:** Network-Panel vor/nach mehrfachem Linie↔Säulen-Wechsel zeigt identische Request-Liste (0 neue Einträge).
- **Regression unberührter Komponenten:** `PriceChartSection` (Kurschart, AnalyzePage) und `PriceComparisonSection` (Kursvergleich) unverändert, da nie `chartType` übergeben. Dashboard-Sparkline (`Sparkline.tsx`) im Browser bestätigt unverändert (eigenständige Komponente, nutzt `MultiLayerChart` nicht). `ChartLayerBuilder.tsx` verifiziert als toter Code (kein Import/Rendering irgendwo im Repo, wie zuvor bereits `TimeSeriesChart.tsx`) – keine Laufzeit-Regression möglich.
- **Dark/Light-Theme:** Säulen-Chart im Hell-Modus im Browser geprüft (ComparePage, „Historischer Umsatz", Quarterly) – Farben/Kontrast/Legende korrekt, keine Regression.
- **Mobile/Tablet:** siehe CHART-007-Nachweise oben (375×812, 768×1024, kein horizontales Scrollen, 4-Firmen-Gruppierung lesbar).
- **Compare-Grenzfall:** 4 Firmen → Säulen möglich; 5. Firma hinzugefügt → Selector verschwindet sofort, Chart fällt automatisch auf Linie zurück; danach wieder auf 4 reduziert → Säulen-Auswahl kehrt automatisch zurück (reine Ableitung aus State + Eligibility, kein Sonderfall-Code).

## 22. Nicht angefasst (bewusst)
`PriceChartSection`, `PriceComparisonSection`, `Sparkline`, `ChartLayerBuilder`, `AdminDashboardPage`,
`TimeSeriesChart` (toter Code) – sie rufen `MultiLayerChart` ohne `chartType` → Default `"line"` →
byte-identisches Verhalten.

## 23. Nicht zweifelsfrei verifizierbar / offen
Exakte recharts-3.8-Bar-Props (`maxBarSize`, negatives Verhalten in `ComposedChart` mit
`allowDataOverflow`) – im Browser gegenzuprüfen. Compare 3 Firmen × Quarterly auf 375px – real messen,
ggf. mobil auf Linie zurückstufen. Phase-2 (%/Margen/Wachstum) zurückgestellt.

## 24. Nachtrag: Tooltip-Deckkraft erhöht (2026-07-22, Betreiber-Feedback)
Der Chart-Tooltip (`ChartTooltip.tsx` – sowohl der Hover-Tooltip als auch das CH-002-Default-Overlay
am neuesten Punkt, beide teilen `containerStyle`) nutzte bislang `theme.colors.panel`
(~72–78 % Deckkraft, dieselbe halbtransparente „Glas"-Fläche wie andere Chart-Flächen) – Linien/Balken
dahinter blieben dadurch sichtbar und beeinträchtigten die Lesbarkeit der Werte. Hintergrund auf
`theme.colors.bgDeepAlt` (nahezu deckend: `#08080a` dunkel / `#fafafa` hell, derselbe Token wie beim
`ChartTypeSelector`-Popover und `SymbolSuggestField`) plus `theme.glass.elevated.shadow` für optische
Abhebung umgestellt. Betrifft **nur** den Chart-Tooltip, nicht `theme.colors.panel` global (Cards,
Modals etc. unverändert). Im Browser in Dark- und Light-Mode verifiziert (computed style: `rgb(8,8,10)`
dunkel / `rgb(250,250,250)` hell, korrekt themenreaktiv).
