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

### [CH-006] Doku, Tests, Build, Verifikation
**Status:** ✅ Umgesetzt (2026-07-19) · **Risiko:** Niedrig
`npx vitest run`: 89/89 grün (70 Bestand + 19 neue). `npm run build`: grün. Commits: CH-001–002 (Teil A), CH-003–005 (Teil B), diese Statusaktualisierung.

**Reihenfolge:** CH-006.1 ✓ → CH-001 ✓ → CH-002 ✓ → CH-003 ✓ → CH-004 ✓ → CH-005 ✓ → CH-006.2 ✓.

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
