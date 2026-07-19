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
**Status:** Offen · **Risiko:** Niedrig
Präsentationalen Kern aus `ChartTooltip.tsx` (Container + Datum-Label + Zeilen inkl. `extractTooltipRows`/`formatCompactNumber`/„–"-Fallback/Currency-Suffix) als benannten Export `ChartTooltipCard({ row, layers, showCurrencyPerRow })` herauslösen. Default-Export (recharts-`content`) behält `!active`-Guards und delegiert. Optik byte-identisch.
**Akzeptanz:** [ ] Hover-Tooltip unverändert (visuell + 70 Bestandstests grün).

### [CH-002] MultiLayerChart: Default-Tooltip am neuesten Punkt
**Status:** Offen · **Risiko:** Mittel
- State `isInteracting` (default false). Desktop: `onMouseMove/onMouseLeave` am `LineChart`; Mobile (`useIsMobile`): `onTouchStart/onTouchEnd` am Wrapper-div (behebt zugleich den klebenden Touch-Tooltip).
- `<Tooltip active={isInteracting ? undefined : false} …>` — `false` unterdrückt recharts-Tooltip im Ruhezustand, `undefined` = normales Hover-Verhalten.
- Bei `!isInteracting && merged.length > 0`: absolutes Overlay im Chart-Wrapper mit `<ChartTooltipCard row={merged[merged.length-1]} …/>`, verankert rechts/vertikal mittig: `right: 16 + (hasRightAxis ? axisWidth : 0); top:50%; translateY(-50%); pointerEvents:none; zIndex:4` (unter Drag-Handles zIndex 5). Kein Überlaufen rechts durch `right:`-Verankerung.
- Wirkt automatisch auf PriceChartSection/PriceComparisonSection/Metrik-Charts. Empty-State unverändert. Compare mit vielen Layern: ggf. maxHeight+overflow (im Browser prüfen).
**Akzeptanz:** [ ] Ohne Hover: Tooltip-Karte mit neuestem Datum+Werten sichtbar [ ] Hover folgt Cursor [ ] Nach Verlassen Rückkehr zum neuesten Punkt [ ] Mobile: sichtbar ohne Tippen, kein klebender Tooltip nach Tap [ ] Drag-Handles weiter bedienbar.

### [CH-003] Reine Helper + Unit-Tests (chartUtils.ts)
**Status:** Offen · **Risiko:** Niedrig
- `localIsoDate(now?)`: lokales `YYYY-MM-DD` (nicht UTC/toISOString — Datumssprung nach Mitternacht DE vermeiden).
- `appendLivePoint(series, livePrice, todayIso)`: null/nicht-finit oder leere Serie → unverändert; spätestes Datum == heute → letzten Wert ERSETZEN; < heute → Punkt ANHÄNGEN; > heute (defensiv) → unverändert. Keine Mutation.
- `computeStartEndTicks(series, format)`: Start/Ende datumsbasiert; `[]` bei <2 gültigen Punkten; Dedupe auf `[ende]` bei Format-Gleichheit, Spannweite 0 oder `|start−ende|/spannweite < 0.25`.
- `formatPriceTick(value, currency)`: `$`-Präfix bei USD (analog LivePriceBadge), `<1000 → toFixed(2)`, `≥1000 → gerundet`.
- `timeRangeLabel(range)`: Lookup über `TIME_RANGE_OPTIONS` („1m"→„1M").
**Akzeptanz:** [ ] Neue vitest-Tests decken ersetzen/anhängen/null/Zukunfts-Skew, Tick-Dedupe, Preisformat, Label-Mapping ab.

### [CH-004] Sparkline: Start/Ende-Y-Achse
**Status:** Offen · **Risiko:** Niedrig-Mittel
Neue optionale Props `showStartEndAxis?: boolean`, `currency?: string | null`. Wenn aktiv und ≥1 Tick: `<YAxis domain={["auto","auto"]} ticks={computeStartEndTicks(...)} tickFormatter={v=>formatPriceTick(v,currency)} width={46} tick={{fontSize:9}} tickLine={false} axisLine={false}>` statt `hide`; margin top/bottom 6 gegen Label-Clipping. `<2` Punkte → weiterhin `null`.
**Akzeptanz:** [ ] Zwei Ticks an korrekten y-Positionen [ ] Kein Clipping [ ] Ohne neue Props verhält sich Sparkline exakt wie bisher.

### [CH-005] DashboardFavoritesSection: FavoriteCard + Live-Punkt + Label
**Status:** Offen · **Risiko:** Mittel
- Karten-Body als `FavoriteCard({ symbol, entry, isLoadingBatch })` extrahieren (Hook-Aufruf!). Dort `useLivePrice(symbol)`; `extended = appendLivePoint(base, price, localIsoDate())`; Sparkline `data={extended} height={48} showStartEndAxis currency={entry.currency}`.
- `computePercentChange(extended)` → Badge und Chart-Ende konsistent (beide inkl. Live-Preis).
- Footer `space-between`: links `timeRangeLabel(entry.range)` (textMuted, nur bei `"rows" in entry`), rechts PercentChangeBadge. Placeholder-Höhe 40→48.
- Batch-`error`-Einträge hinter `"rows" in entry`-Guard → Karte zeigt „–".
**Akzeptanz:** [ ] Chart-Endpunkt == LivePriceBadge-Wert (PYPL-Gegenprobe) [ ] „1M"-Label sichtbar [ ] Fehler-Karten unverändert [ ] Network: keine zusätzlichen current-price-Requests.

### [CH-006] Doku, Tests, Build, Verifikation
**Status:** Teil 1 erledigt (diese Datei ersetzt) · **Risiko:** Niedrig
Nach Abschluss aller Aufgaben: `npx vitest run` (Bestand + neue Tests), echtes `npm run build`; Status hier nachführen; Commits (Teil A / Teil B / Doku) + Push.

**Reihenfolge:** CH-006.1 ✓ → CH-001 → CH-002 → CH-003 → CH-004 → CH-005 → CH-006.2.

## 5. Bekannte Edge-Cases (dokumentiert, bewusst nicht überbaut)

- **Wochenende/Feiertag:** current-price liefert letzten Schlusskurs → wertgleicher flacher Zusatzpunkt mit Wochenend-Datum; keine Börsenkalender-Logik.
- **Zeitzone:** lokales DE-Datum vs. US-Handelstag nach Mitternacht → kosmetische Datumsverschiebung des Live-Punkts.
- **Hybrid-Geräte** (Desktop-Breite + Touch): `isMobile` ist breitenbasiert — Touch auf Desktop-Breite blendet das Overlay bis zum nächsten mouseleave nicht wieder ein (Vereinfachung analog Mobile-Ausblendung der Drag-Handles).
- **Mobile-Overlay-Größe:** ~160px-Karte deckt auf 375px einen Teil des Plots — gewollt; `pointerEvents:none` hält den Chart bedienbar.

## 6. Verifikation (Ende-zu-Ende)

Dev-Server via launch.json (backend FastAPI + frontend Vite); Betreiber-Session im Browser-Tab eingeloggt.
1. `npx vitest run` grün (Bestand + neu), `npm run build` grün (echter Build, nicht nur tsc).
2. **Dashboard:** 2-Tick-Achse ($-Format), „1M"-Label, Chart-Endpunkt == Badge (PYPL), Fehler-Karte „–", Network `current-price` ≤1 Request/Symbol/20s.
3. **AnalyzePage/ComparePage:** Default-Tooltip rechts am neuesten Punkt; Hover folgt; Rückkehr nach Verlassen; Drag-Handles bedienbar; Compare ohne Overlay-Überlauf.
4. **Mobile (375×812):** Default-Tooltip ohne Tippen sichtbar; Tap → Punkt, Loslassen → Rückkehr; kein klebender Tooltip.
5. Screenshots als Beleg.
