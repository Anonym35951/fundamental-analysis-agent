# KORREKTUREN.md — 4 Fixes für ComAnalysis-Frontend

**Status: ✅ Alle 4 Punkte umgesetzt und live im Browser verifiziert (Testkonto,
echte Analysen). `npm run build` fehlerfrei nach jedem Punkt.**

Diese Datei enthielt einen recherchierten Plan für vier UX-Fixes auf der
Analyse- (und teilweise Vergleichs-)Seite. Der Plan-Text unten ist as-is
stehen gelassen (inkl. ursprünglicher Zeilennummern zum Zeitpunkt der
Recherche); die tatsächlich geänderten Dateien sind in den ✅-Vermerken je
Abschnitt gelistet.

---

## 1. "Kursentwicklung anzeigen" nach oben verschieben + default aufgeklappt

**✅ Erledigt.** Umgesetzt in `frontend/src/pages/app/AnalyzePage.tsx`
(Render-Reihenfolge in beiden Branches getauscht) und
`frontend/src/components/charts/PriceChartSection.tsx` (`isOpen` startet
`true`, neuer `useEffect` lädt beim Mount/Symbolwechsel automatisch).
Live verifiziert: Kurschart erscheint oben, offen, mit geladenen Daten, in
Standard- und Individuell-Modus.

**Nur Position und Default-Zustand ändern — keine Funktionslogik anfassen.**

- Datei: `frontend/src/pages/app/AnalyzePage.tsx:798-828`
  Aktuell wird `<PriceChartSection>` **nach** `<AnalyzeResultsDashboard>`
  (Standard-Modus) bzw. **nach** `<CustomAnalysisResultsList>`
  (Individuell-Modus) gerendert. Beide Stellen (Standard-Branch ~Zeile
  800-803, Individuell-Branch ~Zeile 813-816) müssen so umgestellt werden,
  dass `<PriceChartSection>` **zuerst** kommt.
- Datei: `frontend/src/components/charts/PriceChartSection.tsx:28`
  `const [isOpen, setIsOpen] = useState(false);` → `useState(true)`, damit
  der Abschnitt standardmäßig aufgeklappt ist. Der lazy-load-Mechanismus
  (`handleToggle`, Zeile 61-67) lädt Preisdaten nur beim Öffnen — beim
  Default-Open muss der initiale Fetch beim Mount ausgelöst werden (z. B.
  per `useEffect`, der einmalig `load(symbol, range)` aufruft, wenn `isOpen`
  initial `true` ist), sonst bleibt der Chart leer, bis der Nutzer zu-/
  aufklappt.

Kein Eingriff in `MultiLayerChart`, `TimeRangeFilter`, Caching-Logik
(`cacheRef`) oder sonstige Funktionalität nötig.

---

## 2. Analyseausgabe soll modus- und seitenübergreifend sichtbar bleiben

**✅ Erledigt.** Umgesetzt via neuem `AnalyzeWorkspaceProvider`
(`frontend/src/hooks/useAnalyzeWorkspace.tsx` +
`analyzeWorkspaceContextValue.ts` + `useAnalyzeWorkspaceContext.ts`), in
`App.tsx` oberhalb des Routers eingehängt. Hält `analysisTab`,
`currentStandardJobId`, `currentCustomJobId` und ein neues `lastResultKind`.
`AnalyzePage.tsx` zeigt das Ergebnis jetzt anhand von
`displayKind = lastResultKind ?? analysisTab` statt anhand von `analysisTab`
allein. `symbol`/`selectedMode` etc. wurden bewusst NICHT in den Context
verschoben (kein Muss laut Plan) — das Ergebnis hängt nicht davon ab, da
`AnalyzeResultsDashboard`/`CustomAnalysisResultsList` ihr Symbol aus
`result.symbol`/`customResult.symbol` beziehen. Live verifiziert: Ergebnis
bleibt bei Tab-Wechsel sichtbar, übersteht SPA-Navigation zu Konto/Dashboard
und zurück, wird von einer neuen Analyse ersetzt, verschwindet bei Logout.
Vergleichs-Seite hatte das Problem wie vermutet nicht — keine Änderung nötig.

**Ursprüngliches Problem:** Das Analyse-Ergebnis wird aktuell strikt an den aktiven Tab
gekoppelt angezeigt, nicht an "gibt es überhaupt ein Ergebnis":

- `frontend/src/pages/app/AnalyzePage.tsx:798-827` — die Ergebnis-Box
  rendert `result` nur wenn `analysisTab === "standard"`, `customResult` nur
  wenn `analysisTab === "individuell"`. Beide Ergebnisse bleiben technisch im
  State erhalten (`currentStandardJobId`/`currentCustomJobId`, Zeile
  131/172), werden aber nur im "eigenen" Tab angezeigt.
- Gleiche Tab-Kopplung zusätzlich bei: Sticky Bar (Zeile 579-583) und
  Symbol-/Source-Badges (Zeile 746-754).
- **Navigations-Problem:** `symbol`, `analysisTab`, `currentStandardJobId`,
  `currentCustomJobId` etc. sind lokaler `useState` in `AnalyzePage.tsx`
  (Zeile 107-184). `AnalyzePage` ist eine Route (`App.tsx:96`) und wird beim
  Wechsel zu `/app/account`, `/app/support`, `/app/dashboard` von React
  Router **unmounted** — der State geht verloren. Die eigentlichen
  Job-Ergebnisse überleben zwar in `AnalysisJobsProvider` (Context oberhalb
  des Routers, `App.tsx:54-56`), aber `AnalyzePage` verliert nach Remount die
  Job-ID-Pointer, um sie wiederzufinden.

**Lösungsansatz** (Vorbild: `CompareProvider` in
`frontend/src/hooks/useCompare.tsx`, das exakt dieses Problem für die
Vergleichs-Seite bereits löst — dort liegt der komplette Workspace-State in
einem Context oberhalb des Routers, siehe unten):

1. Neuen Context `AnalyzeWorkspaceProvider` anlegen (Pattern wie
   `AnalysisJobsContext`: `frontend/src/hooks/analysisJobsContextValue.ts` +
   `useAnalysisJobs.tsx`), der hält:
   - `analysisTab`, `currentStandardJobId`, `currentCustomJobId`
   - ein neues Feld `lastResultKind: "standard" | "individuell" | null` —
     wird gesetzt, sobald `handleStartAnalysis` bzw.
     `handleStartCustomAnalysis` erfolgreich einen neuen Job startet.
   - optional (für volle State-Kontinuität, kein Muss): `symbol`,
     `selectedMode`, `selectedFrequency`, `selectedDefinition`/
     `adHocMetrics`.
   - **Kein localStorage nötig** — im Gegensatz zu `CompareProvider` reicht
     es, den Context oberhalb der `<Routes>` zu mounten (wie
     `AnalysisJobsProvider` es bereits tut), da die eigentlichen
     Job-Ergebnisse ohnehin schon dort persistent im Speicher liegen. Nur
     die kleinen Pointer/Flags müssen "höher" wandern.
2. Provider in `frontend/src/App.tsx:54-56` neben `CompareProvider`/
   `AnalysisJobsProvider` einhängen.
3. Anzeige-Logik in `AnalyzePage.tsx` von `analysisTab === "..."` auf
   `lastResultKind === "..."` umstellen — an allen drei Stellen (Ergebnis-Box
   Zeile 798-827, Sticky Bar Zeile 579-583, Badges Zeile 746-754). Der
   Tab-Umschalter selbst (`AnalyzeWorkspace`, Standard/Individuell-Buttons)
   bleibt unverändert und steuert weiterhin nur, welches Eingabe-/
   Konfigurationspanel für eine **neue** Analyse sichtbar ist — nicht mehr,
   welches Ergebnis angezeigt wird.
4. Reset-Verhalten: `lastResultKind`/Job-Pointer sollen laut Anforderung
   erst bei Logout oder Start einer neuen Analyse verschwinden. Logout-Event
   existiert bereits als Konvention: `window.dispatchEvent(new
   Event("app:logout"))` in `frontend/src/api/client.ts:47-51`, aktuell nur
   von `CompareProvider` konsumiert (`useCompare.tsx:256-260`).
   `AnalyzeWorkspaceProvider` sollte denselben Event-Listener registrieren
   und seinen State zurücksetzen.

**Vergleichs-Seite:** `ComparePage`/`CompareProvider` hat dieses Problem
nach Recherche **bereits nicht** — der komplette Workspace-State (`metrics`,
`companies`, `frequency`, `hasStarted`) liegt in `CompareProvider`, gemountet
oberhalb des Routers und zusätzlich in `localStorage` gespiegelt
(`useCompare.tsx:18,28-42,68-82`). Es gibt dort auch keinen Standard/
Individuell-Split. → Für die Vergleichs-Seite ist vermutlich **keine
Code-Änderung nötig**, nur eine Verifikation im Rahmen der Tests.

---

## 3. Abstand zwischen Kategorie-Leiste und Kennzahlen-Grid vergrößern

**✅ Erledigt.** `padding: "0 16px 16px"` → `"12px 16px 16px"` in
`MetricCatalogPicker.tsx:266`. Live verifiziert auf Desktop, Tablet (768px)
und Mobile (375px) in allen drei Fundorten (Ad-Hoc-Panel, Definition-Builder,
Compare-Page).

- Datei: `frontend/src/components/customAnalysis/MetricCatalogPicker.tsx:266`
  Der Grid-Container direkt unter der Kategorie-Kopfzeile hat
  `padding: "0 16px 16px"` — der obere Wert ist `0`, wodurch die erste Zeile
  der Kennzahlen-Pills direkt an der Kategorie-Leiste klebt. Fix: oberen
  Padding-Wert auf z. B. `12px` setzen → `padding: "12px 16px 16px"`.
- **Wichtig:** Diese eine Komponente ist die einzige Implementierung dieses
  Musters im gesamten Projekt und wird an drei Stellen wiederverwendet — ein
  einziger Fix behebt alle:
  - `frontend/src/components/customAnalysis/AdHocAnalysisPanel.tsx:72`
    ("Ad-Hoc"/"Einmalige Analyse")
  - `frontend/src/components/customAnalysis/DefinitionBuilder.tsx:41`
    ("Neue Analyse"/"Eigene Analyse erstellen"-Flow)
  - `frontend/src/pages/app/ComparePage.tsx:330` (Kennzahlenauswahl im
    Vergleich)
- Keine weitere Datei muss angefasst werden — es existiert keine zweite/
  duplizierte Implementierung dieses Headers+Grids.
- **Responsiveness beachten:** Bei der Anpassung des oberen Paddings zusätzlich
  auf Tablet- und Handy-Breakpoints prüfen (Dev-Server im Browser auf
  Tablet-/Mobile-Viewport testen bzw. Chrome-DevTools-Gerätesimulation). Der
  Grid-Container nutzt bereits `gridTemplateColumns: "repeat(auto-fit,
  minmax(min(240px, 100%), 1fr))"`, das auf schmalen Screens auf eine
  Spalte kollabiert — sicherstellen, dass der neue obere Abstand dort
  weiterhin stimmig aussieht (nicht zu groß/klein) und nicht mit dem
  bestehenden Verhalten aus `RESPONSIVE.md` kollidiert.

---

## 4. Favoriten erscheinen erst nach Reload in der Sidebar

**✅ Erledigt.** Neuer `FavoritesProvider`
(`frontend/src/hooks/useFavorites.tsx` + `favoritesContextValue.ts` +
`useFavoritesContext.ts`), in `App.tsx` oberhalb des Routers eingehängt.
`AppSidebar.tsx`, `AnalyzePage.tsx` (Stern-Icon) und
`DashboardFavoritesSection.tsx` lesen jetzt alle aus diesem einen Context.
Live verifiziert: Favorit setzen/entfernen erscheint sofort ohne Reload in
Sidebar und Dashboard.

**Ursprüngliche Ursache:** Es gibt keinen geteilten State/Cache für Favoriten. Drei
Komponenten rufen jeweils unabhängig `getFavorites()` einmalig beim eigenen
Mount auf und halten ihr eigenes lokales `useState`:

- `frontend/src/components/layout/AppSidebar.tsx:54,80-84` — Sidebar ist
  dauerhaft gemountet (`AppLayoutInner`), fetcht aber nur beim allerersten
  Mount, nie erneut.
- `frontend/src/pages/app/AnalyzePage.tsx:110,186-190,388-412` —
  `handleToggleFavorite` aktualisiert nur den eigenen `favoriteSymbols`-State
  und ruft `addFavorite`/`removeFavorite` aus `frontend/src/api/favorites.ts`
  auf, ohne die Sidebar zu benachrichtigen.
- `frontend/src/components/dashboard/DashboardFavoritesSection.tsx:23,29-41`
  — gleiches Muster, ebenfalls isoliert.

Es gibt kein SWR/React Query/Zustand im Projekt (verifiziert: keine
entsprechenden Dependencies, keine Treffer für `useSWR(`/`useQuery(`/
`mutate(`). Die etablierte Konvention für geteilten Live-State in diesem
Projekt ist reiner React Context (siehe `AnalysisJobsContext`,
`CompareContext`).

**Lösungsansatz:**

1. Neuen `FavoritesContext`/`FavoritesProvider` anlegen (gleiches Muster wie
   `analysisJobsContextValue.ts` + `useAnalysisJobs.tsx`), der
   `favorites: FavoriteEntry[]`, `isFavorite(symbol)` und
   `toggleFavorite(symbol)` bereitstellt, dahinter `getFavorites`/
   `addFavorite`/`removeFavorite` aus `frontend/src/api/favorites.ts` mit
   optimistischem Update (analog der bisherigen Logik in
   `AnalyzePage.tsx:388-412`).
2. Provider in `frontend/src/App.tsx:54-56` neben `CompareProvider`/
   `AnalysisJobsProvider` einhängen (oberhalb des Routers → übersteht
   Navigation, ein einziger Fetch pro Session reicht).
3. `AppSidebar.tsx`, `AnalyzePage.tsx` (Stern-Icon-Handler) und
   `DashboardFavoritesSection.tsx` auf den neuen Context umstellen statt
   eigener `getFavorites()`-Aufrufe/lokalem State. Da `favoriteSymbols` in
   `AnalyzePage.tsx` durch Punkt 2 ohnehin ggf. in einen Workspace-Context
   wandert, hier direkt konsolidieren: `AnalyzePage` liest `isFavorite`/
   `toggleFavorite` aus dem neuen `FavoritesContext` statt eigenem State zu
   führen.
4. Logout-Reset: analog Punkt 2 optional am `app:logout`-Event anhängen,
   damit Favoriten eines Nutzers nicht in der Session des nächsten sichtbar
   bleiben (gleiches Muster wie `useCompare.tsx:256-260`).

---

## Verifikation (nach jedem Punkt im Browser testen)

Dev-Server: `npm run dev` im `frontend`-Ordner.

1. **Kursentwicklung:** Analyse starten (Standard und Individuell) →
   Ergebnis-Box öffnen → Kurschart muss **oben**, **bereits aufgeklappt**
   und mit geladenen Daten erscheinen, ohne dass ein Klick nötig ist.
2. **Persistenz:** Individuell-Analyse starten → auf Standard wechseln →
   Ergebnis muss weiterhin sichtbar sein → zu Account/Support/Dashboard
   navigieren → zurück zu Analyse → Ergebnis muss weiterhin da sein → neue
   Analyse starten → altes Ergebnis wird ersetzt → Logout/Login → Ergebnis
   muss weg sein. Gleiche Navigations-Prüfung für die Vergleichs-Seite
   (voraussichtlich bereits ok, nur verifizieren).
3. **Kennzahlen-Abstand:** Screenshot-Vergleich vor/nach in allen drei
   Fundorten (Ad-Hoc-Panel, Definition-Builder, Compare-Page) — sichtbarer
   Abstand zwischen Kategorie-Leiste und erster Kennzahlen-Zeile.
4. **Favoriten:** Symbol favorisieren, ohne Reload prüfen, dass es sofort in
   der Sidebar UND im Dashboard-Favoriten-Bereich erscheint; entfernen und
   prüfen, dass es sofort wieder verschwindet — jeweils ohne Reload.

Vor "fertig" `npm run build` im Frontend laufen lassen (nicht nur
`tsc --noEmit`), wie in den bestehenden Projekt-Notizen vermerkt.

---

# Teil 2 — 11 Bugfixes aus einer Frontend-weiten Bug-Suche

**Status: ✅ Alle 11 Punkte umgesetzt und live im Browser verifiziert
(Testkonto, echte Analysen). `npm run build` fehlerfrei nach jeder Gruppe.**
Live/end-to-end verifiziert: #1 (Job leert sich bei Logout, kein
Cross-User-Toast nach Re-Login), #2 (exakte Bug-Zahl 15 Mrd. bei decimals:0
gegen alte/neue Formatierlogik getestet), #3 (Favoriten sofort nach
Client-Side-Login geladen), #6 (neue Firma lädt automatisch bei offenem
Kursvergleich nach, kein Re-Fetch pro Poll-Tick), #8 (Badge zeigt
Ergebnis-Symbol statt Live-Eingabe), #11 (Header aktualisiert sich in einem
zweiten Tab via `storage`-Event ohne Reload). Code-verifiziert (Build +
Konsolen-/Netzwerk-Check ohne Fehler, Logik gegen die Bug-Beschreibung
geprüft), aber ohne exakte Race-Reproduktion in der Browser-Automation:
#4, #5, #7, #9, #10. Zusätzlich alle 4 Teil-1-Punkte erneut gegengetestet
(Kurschart-Position/Default-Open, Ergebnis-Persistenz über SPA-Navigation,
Kennzahlen-Abstand, Live-Favoriten) — keine Regression.

Nach Abschluss von Teil 1 wurde das gesamte Frontend von vier parallelen
Recherche-Agenten nach konkreten Bugs durchsucht. Die Befunde wurden
persönlich im Code verifiziert und ein zusätzlicher Plan-Agent hat alle
vorgeschlagenen Fixes gegengeprüft und verfeinert. Zwei der 11 Bugs sind
eigene Regressionen aus Teil 1 (#3 Favoriten-Race, #4 Kurschart-Race). #1
ist ein echtes Cross-User-Datenleck, #2 eine Datenkorruption bei der
Kennzahlen-Anzeige — beide hatten Priorität.

Ausführlicher Plan mit vollständigem Code für jeden Fix liegt unter
`~/.claude/plans/verschaffe-dir-einen-guten-goofy-wolf.md`. Diese
Zusammenfassung reicht aber aus, um direkt loszulegen.

## Reihenfolge

1. **Vorarbeit:** neues `app:login`-Event einführen (Dispatch in
   `frontend/src/pages/auth/LoginPage.tsx` direkt nach
   `localStorage.setItem("access_token", ...)`, **nicht** zentral in
   `api/client.ts` — nur ein einziger frischer Login-Pfad existiert, anders
   als bei `app:logout`). Wird von #3 und #11 gebraucht.
2. **#1** — Cross-User-Leak (höchste Priorität, unabhängig)
3. **#2, #9, #10** — mechanische Fixes, unabhängig, bündelbar
4. **#3** — Favoriten-Login-Race (braucht `app:login`)
5. **#11** — Header-Reaktivität (braucht `app:login`, geringe Priorität)
6. **#4** dann **#6** — #6 orientiert sich am Muster von #4
7. **#5 und #8 zusammen** — beide in `AnalyzePage.tsx`
8. **#7** — unabhängig

## 🔴 Muss

**#1 — `frontend/src/hooks/useAnalysisJobs.tsx`: Cross-User-Datenleck bei
Logout.** Einziger State-Provider ohne `app:logout`-Listener; `jobs`-Array
(inkl. voller Analyseergebnisse) überlebt den Logout. Fix: `clearAllJobs`
(stoppt alle Polling-Intervalle via `intervalsRef`, setzt `jobs` auf `[]`)
analog `useCompare.tsx`s `clearAll`/`handleLogoutEvent`-Pattern registrieren.
**Nicht** in `AnalysisJobsContextValue` exportieren — bleibt interner
Listener.

**#2 — `frontend/src/components/metrics/metricFormatting.tsx:102`:
Zahlenkorruption.** `scaledValue.toFixed(decimals).replace(/\.?0+$/, "")`
frisst bei `decimals: 0` signifikante Endnullen ganzer Zahlen (z. B.
15.000.000.000 → `"15"`). Betrifft `shares_outstanding`,
`dividend_history`/`analyze_dividend_history`, `asset_value_quality`. Fix:
Regex nur anwenden, wenn `formatted` tatsächlich einen `.` enthält:
```js
let formatted = scaledValue.toFixed(decimals);
if (formatted.includes(".")) {
  formatted = formatted.replace(/\.?0+$/, "");
}
```

## 🟠 Sollte (eigene Regressionen aus Teil 1)

**#3 — `frontend/src/hooks/useFavorites.tsx`: Favoriten laden nach Login
nicht neu.** `FavoritesProvider` fetcht nur einmalig beim App-Mount
(oberhalb des Routers). `LoginPage.tsx` loggt per Client-Navigation ein
(kein Reload) → Fetch-Effekt feuert nie erneut. Fix: Fetch-Logik in
wiederverwendbare Funktion extrahieren, zweiten `useEffect` ergänzen, der
auf das neue `app:login`-Event hört und erneut fetcht.

**#4 — `frontend/src/components/charts/PriceChartSection.tsx`: veraltete
Antworten überschreiben aktuelle Kursdaten.** `load()` prüft beim Auflösen
nicht, ob `symbol`/`range` noch aktuell sind. Fix: monoton steigender
`requestIdRef`-Guard, der `.then()`/`.catch()` nur anwendet, wenn die
Request-ID noch der aktuellsten entspricht (Cache bleibt unconditional
befüllt).

## 🟡 Sollte (bestehende Bugs)

**#5 — `frontend/src/pages/app/AnalyzePage.tsx:272-294`: widersprüchliche
Erfolgs-/Fehlermeldungen.** Vier Effekte setzen Banner je Job-Art, löschen
nie die jeweils andere. **Wichtig:** `pageErrorMessage` wird auch von
generischen Fällen gesetzt (Validierung, Quota, Save) — nicht 1:1 an eine
Job-Art gebunden. Fix: `BannerKind = "standard" | "individuell" | "generic"`
mittracken, Banner nur zeigen wenn `kind === "generic" || kind === displayKind`.

**#6 — `frontend/src/components/charts/PriceComparisonSection.tsx`
(Compare-Seite): lädt bei neuer Firma nicht nach.** Nur
`handleToggle`/`handleRangeChange` lösen `load()` aus. **Wichtig:**
`ComparePage.tsx:412` übergibt bei jedem Render ein neues Array — ein
Effekt mit `[symbols, isOpen]` als Deps würde bei jedem 1,5s-Poll-Tick
feuern. Fix: Effekt auf `symbols.join(",")` (String) statt Array-Referenz
als Dependency, plus denselben `requestIdRef`-Guard wie #4.

**#7 — `frontend/src/hooks/useFavorites.tsx` `toggleFavorite`:
Stale-Closure bei schnellen Doppel-Toggles.** `previous = favorites` aus
Render-Closure erfasst. Fix: `wasFavorite`/`removedEntry` innerhalb des
funktionalen `setFavorites`-Updaters berechnen (läuft synchron gegen
zuletzt eingereihten State); Rollback bei Fehler macht **relativ** nur den
eigenen Toggle rückgängig (erneutes Hinzufügen/Entfernen des einen Symbols),
statt einen absoluten Snapshot zurückzuschreiben, der einen
zwischenzeitlich erfolgreichen zweiten Toggle mit wegwischen würde.

**#8 — `frontend/src/pages/app/AnalyzePage.tsx:745`: Symbol-Badge zeigt
falsches Ticker.** Badge nutzt `normalizedSymbol` (live Eingabefeld), Chart/
Dashboard darunter nutzen `result.symbol`/`customResult.symbol`. Fix:
`resultsBadgeSymbol` ableiten, das je nach `displayKind` `result.symbol`/
`customResult.symbol` bevorzugt und nur auf `normalizedSymbol`
zurückfällt, wenn (noch) kein Ergebnis existiert.

## 🟢 Kann warten (Politur)

**#9 — `frontend/src/components/ui/InfoTooltip.tsx:143-199`:**
Exit-Fade-Animation greift nie, da die komplette `AnimatePresence` (nicht
nur ihr Kind) an `isOpen` hängt. Fix: Portal + `AnimatePresence` immer
rendern, nur das `motion.div`-Kind an `isOpen` binden (plus stabilen
`key`).

**#10 — `frontend/src/components/admin/CustomerListView.tsx:106-119`:**
Tabellenzeilen nur per Maus erreichbar (`onClick` auf `<tr>`, kein
Keyboard-Äquivalent). Fix: gleiches `role="button"` + `tabIndex={0}` +
`onKeyDown`-Idiom wie bereits in `InfoTooltip.tsx:101-127` etabliert, auf
die `<tr>` anwenden.

**#11 — `frontend/src/components/layout/Header.tsx:9`:** Login-Status wird
nur einmal aus `localStorage` gelesen. Nur cross-tab relevant (Header wird
bei Login/Logout im selben Tab durch Layout-Wechsel immer neu gemounted).
Fix: `useState` + `useEffect`, der auf `app:login`/`app:logout`/`storage`
hört und `hasToken` neu liest.

## Verifikation Teil 2

Nach jeder Gruppe `npm run build`, dann im Browser (Testkonto) prüfen — je
ein konkretes Test-Szenario pro Punkt steht im ausführlichen Plan unter
`~/.claude/plans/verschaffe-dir-einen-guten-goofy-wolf.md`. Am Ende
zusätzlich die 4 Punkte aus Teil 1 erneut gegentesten, da mehrere Fixes
dieselben Dateien anfassen (`AnalyzePage.tsx`, `useFavorites.tsx`,
`PriceChartSection.tsx`).
