# EVOLVING – Mobile Performance, Landingpage und TTM

> Planungsdokument, erstellt am 2026-07-16 nach vollständiger Code-Analyse (drei parallele Explorationen + Architektur-Review).
> Die frühere EVOLVING.md wurde nach vollständiger Umsetzung gelöscht (Commit 51b9032); dieses Dokument ist ein Neuanfang.
> Aufgaben-IDs starten bei **EV-100**, um Kollisionen mit der gelöschten Alt-Datei auszuschließen.
>
> **Einstufungslegende für alle Aussagen:** `[Bestätigt]` = im Code mit Datei:Zeile belegt · `[Sehr wahrscheinlich]` = starke Codeindizien, noch nicht gemessen · `[Hypothese]` = plausibel, unbelegt · `[Offen]` = muss während der Implementierung ermittelt werden.

## 1. Ziel und Hintergrund

Drei Vorhaben:

1. **Mobile Performance:** Die App ist auf Smartphones deutlich zu langsam (sekundenlang blockierte UI, zeitversetzt erscheinende Elemente, verzögerte Interaktionen). Die Ursachen sind jetzt im Code eingegrenzt (siehe §6) und werden in kleinen, messbaren Schritten behoben.
2. **Landingpage-Kommunikation:** Wording und Nutzenkommunikation werden geschärft; drei sachliche Diskrepanzen (SEC-Claim, Kursaktualität, „Analysen" vs. „Einheiten") werden zuerst korrigiert.
3. **TTM:** Die Periodenauswahl `annual`/`quarterly` wird um `ttm` (Trailing Twelve Months) als dritte Option erweitert (Produktentscheidung: Variante B) und `ttm` wird Frontend-Default.

Umsetzendes Modell: Sonnet 5, Aufgabe für Aufgabe gemäß §14, in der Reihenfolge aus §13/§15.

## 2. Nicht verhandelbare Anforderungen

- **Annual darf sich fachlich nicht ändern** — keine Formel-, Datenquellen-, Ergebnis-, Cache- oder API-Vertragsänderung. Beweis durch Golden-Master-Tests (§17), die vor jedem TTM-Schritt grün sein müssen.
- **Quarterly bleibt erhalten und liefert unveränderte Ergebnisse** (§18).
- **Keine neuen Dependencies:** package.json/Lockfiles bleiben unangetastet (Frontend und Backend).
- **Keine Datenbankmigration:** `analysis_history.frequency` ist `String(20)` nullable und nimmt „ttm" migrationsfrei auf ([api/models/analysis_history.py:29]).
- **Backend-Query-Defaults bleiben `"annual"`** (API-Vertrag für bestehende Clients); nur das Frontend sendet explizit `frequency=ttm`.
- **Neutralitäts-Leitplanke:** reine Information, keine Anlageberatung; kein Landingpage-Claim ohne Beleg im Produkt; keine Buzzwords/Superlative.
- **Desktop-Verhalten unverändert:** Partikelstrahl bleibt auf Desktop (Produktentscheidung); Intro-Overlay-Timing (`AppLayout.tsx:58-98`), Provider-Reihenfolge in `App.tsx`, Sentry-Konfiguration und Fonts werden nicht angefasst.
- **Vor jedem „fertig": echtes `npm run build`** im Frontend (nicht nur `tsc --noEmit`).

## 3. Analysierte Projektarchitektur

`[Bestätigt]` (alle Punkte aus package.json, requirements.txt und Code):

| Bereich | Technologie / Fundstelle |
|---|---|
| Frontend-Framework | React 19.2 + Vite 7 (`frontend/package.json`), TypeScript 5.9, kein SSR — reine Client-SPA |
| Routing | react-router-dom 7, `BrowserRouter` in `frontend/src/App.tsx` |
| State-Management | React-Context-Provider: `CompareProvider > AnalysisJobsProvider > AnalyzeWorkspaceProvider > FavoritesProvider > ThemeModeProvider` (`App.tsx`); kein Redux/Zustand/React Query |
| API-Client | eigener fetch-Wrapper `frontend/src/api/client.ts` (kein Caching, keine Dedup) |
| Charts | recharts 3.8 (14 `ResponsiveContainer`-Nutzungen) |
| Animation/3D | framer-motion 12, three 0.184 + @react-three/fiber (nur IntroOverlay), Canvas-2D-Partikel (`ParticleBeamBackground.tsx`) |
| Onboarding | react-joyride (`components/onboarding/AppTour.tsx`) |
| Monitoring | @sentry/react, ohne Replay/Tracing (`main.tsx:9-14`) |
| Analytics | Cloudflare Web Analytics Beacon, consent-gated (`lib/webAnalytics.ts`); Backend-Events via `api/services/event_service.log_event` → Tabelle `product_events` |
| Backend | FastAPI (`api/`: routes, services, schemas, models, crud, jobs), Auth + Stripe-Billing |
| Analyse-Engine | `agent/`: `DataLoader.py`, `Model.py` (~5300 Z.), `AgentAction.py` (7 Analysemodi), `AgentOrchestrator.py`, `growth_math.py`, `industry_multiples.py` |
| Datenanbieter | SEC EDGAR companyfacts (Fundamentaldaten primär, `agent/data_sources/sec_source.py`), Alpha Vantage (Statements, `DataLoader.py:87-88,459-546`), Yahoo/yfinance (Kurse, `.info`, Historie, Dividenden, `DataLoader.py:6,143-146`), FRED (Makro) |
| Datenbank/ORM | Postgres + SQLAlchemy + Alembic (`alembic/`) |
| Caching | Datei-Cache `cache/` (>2300 Dateien), Key = `{symbol}_{data_type}.json` (`DataLoader.py:3208`), TTL-Staffelung 600 Tage/6 h/10 s (`DataLoader.py:56-58`), B2-Objekt-Spiegelung (`agent/cache_object_storage.py`) |
| Tests | pytest (`agent/tests/`, `api/tests/`), vitest (Frontend, wenige Tests) |
| Build/Deploy | Vite-Build ohne Optimierungskonfig (`vite.config.ts`), Frontend + Backend auf Render |

## 4. Relevante Datenflüsse

### 4.1 Mobile Rendering und Datenladen

**Initialer Seitenaufruf (jede Route):** Ein einziges JS-Haupt-Bundle (1,19 MB roh / **349 KB gzip**, `frontend/dist/assets/index-*.js`) wird geladen und geparst — es enthält alle 22 Seiten, recharts, framer-motion und react-joyride, weil `App.tsx:3-43` alles statisch importiert und `vite.config.ts` keine Chunk-Konfiguration hat. Einziger Lazy-Chunk: IntroOverlay/Three.js (891 KB), auf Mobile nie geladen (`AppLayout.tsx:72-77`). `[Bestätigt]`

**Datenladeplan Dashboard (eingeloggt, wichtigste Seite):**

| # | Request | Auslöser | Abhängigkeit | Payload | Cache | Blockierend? | Sichtbare Wirkung | Optimierung |
|---|---|---|---|---|---|---|---|---|
| 1 | `GET /me` ×4+ parallel | DashBoardPage:98, AppSidebar:69, EmailVerificationBanner:18, useAppTour:65 | Token | klein | keiner | Inhalte je Konsument | Begrüßung/Sidebar/Banner erscheinen einzeln | EV-112: 1 Request statt 4 |
| 2 | `GET /favorites` | FavoritesProvider (`useFavorites.tsx:14-24`) | Token | klein | Context (dedupliziert, ok) | Favoriten-Sektion | Favoritenliste | — |
| 3 | `GET /analysis-history` | DashBoardPage nach /me | seriell nach 1 | mittel | keiner | Historien-Karte | Karte poppt nach | EV-115 Skeleton |
| 4 | `GET /custom-metrics-catalog` | DashBoardPage | parallel | mittel | keiner | nein | — | — |
| 5 | `GET /price-history/batch` | DashboardFavoritesSection:39 | nach 2 | mittel (1m/3m-Sparklines) | Server 15 min | Sparklines | Charts poppen nach | EV-115 |
| 6 | `GET /current-price/{sym}` × ~2N alle 20 s | je LivePriceBadge-Instanz (`useLivePrice.ts:61`); Sidebar + Dashboard pollen dieselben Favoriten doppelt | nach Mount | klein, aber viele | Server 12 s | nein | Preise poppen einzeln ein | EV-113: N statt 2N, geteilter Store |

**Layout Shifts / zeitversetztes Erscheinen:** Kaskade aus Bundle-Parse → 4×/me → Historie/Favoriten → Batch-Preise → Einzel-Preis-Polls, jeweils mit eigenem `isLoading`-State ohne reservierte Flächen; `LivePriceBadge` wechselt von „…" auf Wert ohne feste Breite. Parallel konkurriert der Partikel-Canvas (bis 520 Partikel, rAF-Loop) um den Main Thread. `[Bestätigt]` für die Mechanik, `[Sehr wahrscheinlich]` als Ursache der Nutzerbeobachtung.

**Aktive Live-Objekte pro App-Seite (mobil):** Partikel-Canvas-rAF (Landing/Dashboard/Account/Billing/Auth), 1 framer-motion-Infinite-Pulse **pro** LivePriceBadge (`LivePriceBadge.tsx:25-29`), 1 20s-Interval pro Badge, 10-min-Keep-Alive (`SessionTimeoutWatcher.tsx:95`, unkritisch), Job-Polling nur bei laufenden Jobs (`useAnalysisJobs.tsx:107`, Cleanup ok). Cleanup-Funktionen sind überall vorhanden `[Bestätigt]` — das Problem ist Menge und fehlende Mobile-Reduktion, nicht Leaks.

### 4.2 Landingpage und Conversion

- Route `/`: mit `access_token` Redirect auf `/app/dashboard`, sonst `<LandingPage />`; zusätzlich `/landing` (`App.tsx:66-74`). Layout: `PublicLayout` = Header + Main + Footer + AmbientBackground.
- Sektionsfolge: Hero (94svh) → Value („Warum ComAnalysis") → How-it-works → Final-CTA → Footer. Komponenten unter `frontend/src/components/landing/`.
- CTA-Routing: `/register` (2×), `/login` (2×), `/pricing` (1× inline + Header/Footer). Header zustandsabhängig via `hasToken` (localStorage).
- Analytics: nur Cloudflare-Pageviews nach Consent (`CookieConsentBanner.tsx`, localStorage `analytics_consent`); **kein** CTA-/Funnel-Event. Backend loggt `user_registered` etc. in `product_events` — Brücke zur Landingpage fehlt (EV-124).
- Registrierung: `/register` → `POST /register` → `user_registered`-Event (ohne Quelle).

### 4.3 Annual-, Quarterly- und TTM-Datenfluss

1. **Auswahl:** `FrequencyToggle.tsx:3-11` (hartkodiert `"annual" | "quarterly"`). AnalyzePage-Default `"annual"` (State, nicht persistiert; Rerun stellt `job.frequency` aus DB-History wieder her, `AnalyzePage.tsx:349,367-370`). Compare persistiert in localStorage `compare_workspace_v2` (`useCompare.tsx:18,30-40`). Custom-Definitionen speichern frequency **pro Metrik** im JSON-`metrics`-Feld (`api/models/custom_analysis_definition.py:29-34`).
2. **Request:** `frontend/src/api/analysis.ts:100-103` (`?frequency=`), `customAnalysis.ts:23,43`, `dataSource.ts:12-14`. Typen: `types/customAnalysis.ts:1`, `hooks/compareContextValue.ts:4`.
3. **Validierung:** Routen nehmen `frequency: str = "annual"` (`api/routes/analyze.py:138-244`, `metric_routes.py:94-396`, `custom_analysis.py:65-464`) — **keine Pydantic-Enum**. Zentrale Enum nur `api/services/metric_catalog.py:68-71` (`FREQUENCY_PARAM`, enum_values=["annual","quarterly"]). Harte Guards `if frequency not in ["annual","quarterly"]` an ~30 Stellen (DataLoader.py ×13, Model.py ×9, sec_source.py ×6, AgentAction.py ≥1). `[Bestätigt]`
4. **Anbieter:** SEC companyfacts (annual/quarterly; quarterly für 20-F-Foreign-Issuer bewusst blockiert, `sec_source.py:98-106,217-225,474-482`), Alpha Vantage `annualReports`/`quarterlyReports` (`DataLoader.py:505-510`), yfinance (Kurse; `trailingEps`/`trailingPE` nur als Fallbacks, `DataLoader.py:611,752-841,2632`). **Kein Anbieter liefert TTM-Statements.** `[Bestätigt]`
5. **Normalisierung:** `DataLoader.get_fundamental_data`/`get_stock_financials`/`get_balance_sheet`; `growth_math.py:22-65` (CAGR über Datumsdifferenz, period-agnostisch).
6. **Berechnung:** `Model.py` — siehe TTM-Matrix §9. **Zentraler Befund:** `frequency="annual"` bedeutet für aktuelle Flow-Kennzahlen bereits TTM (KUV: Summe letzter 4 Quartale, `Model.py:481-490`; ROE: `Model.py:540-556`), und **alle** `calculate_historical_*`-Reihen laden fest quarterly und rollen mit `.rolling(window=4, min_periods=4).sum()` zu TTM (`Model.py:2160,2329,2501`), unabhängig von der Nutzerwahl. 93 `_ttm_`-Cache-Dateien existieren (z. B. `cache/AAPL_historical_ebit_ttm_AAPL.json`). Umstellung erfolgte am 2026-07-11 („LAUNCH.md Block 5"). Nur Dividenden (`resample("YE")`, `Model.py:245`), AAGR (`Model.py:1545`) und Payout nutzen echte Jahresdaten. `[Bestätigt]`
7. **Analyse/Scoring:** 7 Modi in `AgentAction.py` (Zeilen 20/295/494/773/1196/1380/1496); Schwellen hartkodiert und period-unabhängig (z. B. Payout 75.0, P/TBV 1.2). Vollanalyse: `api/routes/full_analysis.py:30-95` mit festen frequencies-Listen pro Modus (Wachstum/Zykliker/Turnarounds `["annual","quarterly"]`, Rest `["annual"]`).
8. **Speicherung:** `analysis_history` mit `frequency` String(20) nullable + `result_snapshot` JSON (`api/models/analysis_history.py:26-52`); Custom-Definitionen JSON; Compare nur localStorage; Favoriten ohne Periode.
9. **Darstellung:** Ergebnis-Key `"<NAME>|<frequency>"` (`analyze.py:271`, `types/analysis.ts:39-45`), `AnalysisFrequencyPanel.tsx:23` zeigt `frequency.toUpperCase()`.
10. **Export:** existiert nicht (Produktentscheidung: kein Export).

## 5. Aktueller Zustand

### 5.1 Mobile Performance
Ein monolithisches Bundle, ungedrosselte Partikel-Animation, redundante Requests, Poll-Vervielfachung, keine Memoisierung, keine Skeletons — Details in §6. Messwerte: **noch nicht erhoben** (EV-100), nur Bundle-Größen sind gemessen.

### 5.2 Landingpage-Kommunikation
Kernbotschaft „Nachvollziehbarkeit" (H1 „ALLE FUNDAMENTALDATEN. EINE QUELLE. JEDE ZAHL NACHVOLLZIEHBAR.") ist stark und belegbar. Schwächen: drei Wahrheits-Diskrepanzen (§8), SEO-Basics fehlen (`lang="en"`, statischer Title, keine meta/OG, Vite-Favicon), keinerlei CTA-Messung.

### 5.3 Periodenmodell
Zweiwertig `annual|quarterly` als String-Literale, verstreute Guards, gemischte Annual-Semantik (teils TTM, teils echtes Geschäftsjahr) — siehe §4.3 Punkt 6.

### 5.4 Annual-Analysen
Funktionieren zufriedenstellend; Default überall. Semantik: für aktuelle Flow-Kennzahlen und alle historischen Reihen faktisch TTM, für Dividenden/AAGR/Payout echtes Jahr. Diese Ist-Semantik wird **nicht** verändert, nur dokumentiert (EV-138).

### 5.5 Quarterly-Analysen
Echte Einzelquartale `[Bestätigt]` — Beleg: `cache/AMD_historical_income_statement_quarterly.json`, `grossProfit` oszilliert saisonal (2026-03: 5,416 Mrd; 2025-12: 5,577; 2025-09: 4,780; 2025-06: 3,059 …), kein YTD-Muster mit Q1-Reset. Kommentar `Model.py:2151-2153` bestätigt Einzelquartal-Semantik. Bekannte, bestehende fachliche Inkonsistenz: Einzelquartalswerte werden gegen dieselben (12-Monats-)Schwellen geprüft wie Annual/TTM — wird dokumentiert, nicht angefasst.

### 5.6 Custom Analysis Modes
Definitionen speichern frequency pro Metrik im JSON (`custom_analysis_definition.py:29-34`); gespeicherte Definitionen mit `"quarterly"` bleiben durch Variante B unverändert gültig. Free-Limit: 1 Definition, Pro: unbegrenzt.

### 5.7 Vergleichsanalysen
`ComparePage.tsx` + `useCompare.tsx`; frequency in localStorage (`compare_workspace_v2`), `bucketMode = quarterly ? "quarter" : "year"` (`ComparePage.tsx:50-53`). Nicht DB-persistiert.

## 6. Bestätigte Probleme und Beweise

| # | Problem | Fundstelle | Technischer Zusammenhang | Nutzerwirkung | Einstufung |
|---|---|---|---|---|---|
| P1 | Kein Route-Code-Splitting; 349 KB gzip Initial-JS | `App.tsx:3-43`, `vite.config.ts:1-13`, `AppLayout.tsx:13` (joyride statisch) | Parse/Compile blockiert Mobile-Main-Thread mehrere 100 ms bis >1 s | „UI sekundenlang blockiert" beim ersten Laden | Bestätigt (primär) |
| P2 | Partikel-Canvas ohne Mobile-Reduktion, bis 520 Partikel, full-page rAF | `ParticleBeamBackground.tsx:20-26,95-97,132-153`; Mounts: LandingPage:127, DashBoardPage:207, AccountPage:400, BillingPage:94, AuthLayout:25 | Dauerhafte Canvas-Neuzeichnung bei dpr≤2 konkurriert mit Datenladen | Ruckeln, verzögerte Interaktion auf Kernseiten | Bestätigt |
| P3 | 4+ parallele `/me`-Requests pro App-Seite | `DashBoardPage.tsx:98`, `AppSidebar.tsx:69`, `EmailVerificationBanner.tsx:18`, `useAppTour.ts:65` (+4 seitenspezifisch) | Kein Cache/Dedup im fetch-Wrapper | Netzwerk-/Serverlast, gestaffeltes Erscheinen | Bestätigt |
| P4 | Live-Preis-Poll pro Badge (~2N Requests/20 s) + Infinite-Pulse pro Badge | `useLivePrice.ts:10,61`, `LivePriceBadge.tsx:15,25-29`, `AppSidebar.tsx:319,566` | Sidebar + Dashboard pollen dieselben Favoriten unabhängig; N Animation-Ticker | Dauerlast, Preise poppen einzeln | Bestätigt |
| P5 | 0× `React.memo` projektweit; Chart-Transformation ohne useMemo | grep-Ergebnis; `TimeSeriesChart.tsx:50` | Parent-Re-Renders propagieren in teure Teilbäume (Charts, Badges) | verzögerte Reaktionen | Sehr wahrscheinlich |
| P6 | Async-Kaskaden ohne Skeletons/reservierte Flächen | `DashBoardPage.tsx:98 ff.`, `DashboardFavoritesSection.tsx:29-57`, `LivePriceBadge.tsx:31` | Unabhängige isLoading-States, keine festen Höhen/Breiten | „Objekte erscheinen zeitversetzt", Layout Shifts | Sehr wahrscheinlich |
| P7 | Lange Kursreihen (5y/max, >1000 Punkte) ungefiltert in recharts | `api/marketData.ts:23,38` | Render-/Tooltip-Kosten pro Chart | träge Charts | Möglich (Messung EV-100) |
| — | **Kein Befund (positiv):** Three.js-Intro mobil aus (`AppLayout.tsx:72-77`), Sentry ohne Replay (`main.tsx:9-14`), 1 Font mit swap, `public/` leer, Provider-Values memoisiert (`useFavorites.tsx:90-93`), Cleanup-Disziplin gut | | | | Bestätigt |

**Landingpage-Diskrepanzen:** siehe §8 (D1–D3). **TTM-Befunde:** siehe §4.3/§5.5.

## 7. Performance-Baseline

Noch **keine** Messwerte erhoben — alle Zielwerte in §14 sind relativ zur Baseline aus EV-100 definiert. Messkonzept:

- **Umgebung:** Chrome DevTools Lighthouse Mobile (Moto G Power Preset, 4× CPU-Throttling, Slow 4G) gegen lokalen Production-Build (`npm run build` + `vite preview`); zusätzlich, soweit verfügbar, ein reales iPhone (Safari) und ein leistungsschwaches Android (Chrome) gegen die Render-Produktions-URL.
- **Szenarien:** kalter/warmer Cache × ausgeloggt/eingeloggt × Free/Pro × wenige/viele (≥10) Favoriten × frisch nach Login vs. nach längerem Verbleib; Hoch-/Querformat stichprobenweise.
- **Seiten (Priorität nach Nutzeraufkommen/Langsamkeit/Geschäftsrelevanz):** 1. `/app/dashboard`, 2. `/app/analyze` (mit großer Analyse), 3. `/` (Landing), 4. `/app/compare`, 5. `/login`+`/register`, 6. `/app/account`, `/app/billing`; Navigation/Sidebar/Mobile-Nav im Rahmen der Seitenmessungen.
- **Kennzahlen:** FCP, LCP, INP, CLS, TTFB, TBT, JS-Ausführungszeit, Anzahl Long Tasks, initiale Transfermenge, initiale JS-Größe, Anzahl initialer Requests (inkl. gezählter `/me`- und `/current-price`-Requests im Network-Tab), JS-Heap nach 2 min Nutzung, Zeit bis Charts benutzbar, Dauer Seitenwechsel Dashboard→Analyze.
- **Protokoll:** Ergebnistabelle wird nach EV-100 hier eingefügt; jede Optimierung (EV-110…EV-116) protokolliert Vorher/Nachher in ihrem Task.

## 8. Content-Inventur der Landingpage

Quelle: `frontend/src/pages/public/LandingPage.tsx` (827 Z.). Format: Abschnitt / aktuelle Kernaussage / beabsichtigte vs. tatsächliche Wirkung / Problem / empfohlene Botschaft / Beleg / primärer CTA.

| Abschnitt (Zeile) | Aktuelle Kernaussage (wörtlich) | Beabsichtigt | Tatsächliche Wirkung | Problem | Empfohlene neue Botschaft (Richtung) | Beleg im Produkt | Primärer CTA |
|---|---|---|---|---|---|---|---|
| Hero H1 (153-166) | „ALLE FUNDAMENTALDATEN. EINE QUELLE. JEDE ZAHL NACHVOLLZIEHBAR." | Differenzierung Transparenz | Stark, aber drei Statements konkurrieren | Etwas abstrakt; „eine Quelle" kollidiert mit D1 | Nachvollziehbarkeit schärfen: Ergebnis zuerst („Du siehst zu jeder Kennzahl Quelle und Rechenweg") | SourceBadge/Formeln in AnalysisPage, Glossar | „Kostenlos starten" → /register |
| Hero-Subtext (176-179) | „bündelt SEC-Originaldaten und transparent berechnete Kennzahlen … Interpretation bleibt bei dir." | Vertrauen + Neutralität | gut | „SEC-Originaldaten" pauschal (D1) | Fundamentaldaten=SEC präzisieren, Kursdatenquelle ergänzen | DataLoader-Quellen | — |
| Hero-Checkliste (215-222) | „Direkt aus SEC-Filings — nicht aus Dritt-Aggregatoren" / „Jede Formel offen einsehbar" / „Keine Anlageberatung…" | Glaubwürdigkeit | Punkt 1 nur teilweise wahr | **D1** (s. u.) | „Fundamentaldaten direkt aus SEC-Filings. Kursdaten von etablierten Marktdatenanbietern." | `DataLoader.py:6,87-88,165,189` | — |
| Hero-CTA (189-210) | „Kostenlos starten — 50 Analysen/Monat inklusive" | Conversion | klar | **D3**: Backend zählt „Einheiten" | Terminologie vereinheitlichen (EV-121c) | `dependencies.py:143-167`, `pricingPlans.ts` | /register |
| Methoden-Panel (56-96) | 4 rotierende Modi-Benefits + Workflow | Produkt zeigen | gut, konkret | Rotation auf Mobile ggf. unruhig (nur 5s) | beibehalten, Texte im Copy-Deck schärfen | 8 Modi in `AnalyzePage.tsx:52-96` | — |
| Duo-Karten (262-277) | „Kennzahlen-Engine" / „Eigene Logik" | Sekundärfeatures | ok | „in Sekunden" grenzwertig (Vollanalyse dauert länger) | messbaren Anspruch entschärfen oder belegen | Analyse-Jobdauer | — |
| Stat-Readouts (293-295) | „Methoden: 8 vordefiniert · Builder: Eigene Logik inklusive · Plattform: Account & Billing integriert" | Substanz | „Account & Billing integriert" ist kein Nutzervorteil | schwächster Readout | durch nutzerrelevanten Fakt ersetzen (z. B. Vergleich mehrerer Unternehmen, historische Multiples) | ComparePage, historische Charts | — |
| Value (303-324) | „Eine Plattform statt drei Werkzeuge" + 3 Cards | Bündelungs-Nutzen | gut | Card 3 („kein Prototyp, sondern ein nutzbares Produkt", „Stripe-Billing") argumentiert defensiv/intern | Cards auf reale Kern-Features mappen: 8 Modi / Custom-Builder / Vergleich + historische Charts; Favoriten als Nebensatz | AnalyzePage, customAnalysis/, ComparePage | sekundärer CTA sinnvoll |
| How-it-works (333-346) | „Vom Login zur strukturierten Analyse", 4 Schritte | Einstiegshürde senken | Schritt 1 „Produkt und Methoden verstehen" ist kein Arbeitsschritt | Flow-Abgleich nötig | realen Flow abbilden: Symbol suchen → Modus/eigene Logik wählen → Ergebnis mit Quellen lesen → vergleichen/speichern | AnalyzePage-Flow | — |
| Final-CTA (365-406) | „Bereit für deine erste strukturierte Analyse?" | Conversion | ok | „erlebe, wie … aussehen kann" schwach | konkretes Ergebnisversprechen (belegbar) | — | /register |
| Footer | 7 Links (Glossar, Preise, Legal) | Pflicht | ok | — | unverändert | — | — |

**Diskrepanzen (Wahrheitsschulden, Fix vor jeder Stilarbeit — EV-121):**
- **D1** „nicht aus Dritt-Aggregatoren": Kurse/Kurshistorie kommen von Yahoo/yfinance, Statements teils Alpha Vantage (`agent/DataLoader.py:6,87-88,143-146,459-546`). Fundamentaldaten-Claim stimmt (SEC/FRED), Pauschalaussage nicht.
- **D2** Kursaktualität nirgends kommuniziert: Server-Cache 12 s (`metric_routes.py:41`), yfinance-Kurse für viele Börsen ~15 min verzögert; weder „Echtzeit"-Overclaim noch Delay-Hinweis vorhanden → Hinweis ergänzen.
- **D3** „50 Analysen/Monat" (Hero) vs. „50 Analyse-Einheiten" (Pricing) vs. Backend-`units` (`dependencies.py:143-167`) → Terminologie vereinheitlichen, vorher fachlich klären, wie viele Einheiten eine Vollanalyse verbraucht.

**Feature→Nutzen-Übersetzung (Basis für Copy-Deck EV-122):**

| Feature | Funktionaler Nutzen | Emotionaler Nutzen | Ergebnis |
|---|---|---|---|
| Nachvollziehbare Kennzahlen (Quelle+Formel) | Jede Zahl prüfbar bis zum SEC-Filing | Kein Blackbox-Misstrauen | Eigenständige, belastbare Einschätzung |
| 8 vordefinierte Analysemodi | Strukturierte Auswertung ohne Setup | Nicht bei null anfangen | Vollständiges Unternehmensbild in einem Durchlauf |
| Eigene Analyselogik (Builder) | Bewertung nach eigenen Kriterien | Unabhängig von starren Standardanalysen | Wiederverwendbare persönliche Analysemethode |
| Vergleichsanalyse | Mehrere Unternehmen nebeneinander | Schluss mit Tabellen-Jonglage | Direkter Kriterienvergleich in einer Ansicht |
| Aktuelle Kurse + historische Multiples | Bewertung im historischen Kontext | Sicherheit durch Einordnung | Von Rohdaten zu strukturierter Einschätzung ohne Tool-Wechsel |

**Zielgruppe (Standardannahme, s. §11):** private Selbstentscheider / langfristige Einzelaktien-Anleger mit Grundkenntnissen in Fundamentalanalyse.

**Kommunikationshierarchie (empfohlen, bestehende Sektionsstruktur bleibt — nur Texte):** 1. Ergebnis (nachvollziehbare Einschätzung) → 2. Problem (verstreute Quellen, Blackbox-Kennzahlen) → 3. Differenzierung (Quelle+Rechenweg je Zahl) → 4. eigene Logik → 5. Vergleich → 6. Kurse+Historie → 7. gebündelte Arbeitsschritte → 8. Registrierungsgrund (Free-Kontingent) → 9. CTAs wie heute (Hero primär Registrierung, Final-CTA Wiederholung, Login sekundär). Kein Struktur-/Layout-Umbau nötig.

## 9. TTM-Kennzahlenmatrix

Grundlage: vollständiger Nutzer-Katalog `api/services/metric_catalog.py:208-335` + intern genutzte Methoden. **v1-Prinzip: `ttm` = Delegation auf den heutigen Annual-Codepfad** (der für diese Kennzahlen bereits TTM rechnet) — keine neue Numerik. Kategorien: **F**=Flow (4Q-Summe), **PIT**=Point-in-Time (letzter Quartalsstichtag), **R**=Ratio aus TTM-Komponenten, **HIST**=historische Reihe (ist bereits TTM-Rolling für jede Nutzerwahl), **N/A**=kein sinnvolles TTM in v1.

| Interner Name | Anzeigename | Kat. | Annual heute | Quarterly heute | TTM-Regel v1 | TTM? | Risiko |
|---|---|---|---|---|---|---|---|
| calculate_kuv | KUV (P/S) | R(F) | TTM-Umsatz (4Q-Summe, Model.py:481-490) | letztes Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_ev_to_sales | EV / Sales | R(F) | TTM-Komponenten | Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_ev_to_ebit | EV / EBIT | R(F) | TTM | Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_ev_to_ebitda | EV / EBITDA | R(F) | TTM | Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_price_to_ebit | Price / EBIT | R(F) | TTM | Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_price_to_freeCashflow | Price / FCF | R(F) | TTM | Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_roe | ROE | R(F/PIT) | TTM-Nettogewinn (Model.py:540-556) / EK | Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_ROIC | ROIC | R(F/PIT) | durchgereicht | durchgereicht | = Annual-Pfad; Nenner-Definition (Ø vs. Stichtag) in EV-131 verifizieren | ✓* | mittel |
| calculate_cashflow_margin | Cashflow-Marge | R(F) | durchgereicht | Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_net_debt_to_ebitda | Net Debt / EBITDA | R(PIT/F) | durchgereicht | Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_interest_coverage_ratio | Interest Coverage | R(F) | durchgereicht | Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_inventory_to_revenue_ratio | Inventory / Revenue | R(PIT/F) | durchgereicht | Quartal | = Annual-Pfad | ✓ | niedrig |
| calculate_debt_to_equity | Debt-to-Equity | PIT | Bilanz je Frequenz | Bilanz Quartal | letzter Quartalsstichtag (= heutiges Ratio-Verhalten) | ✓ | niedrig |
| calculate_cash_to_market_cap | Cash / Market Cap | PIT | durchgereicht | Quartal | letzter Quartalsstichtag | ✓ | niedrig |
| calculate_current_netCurrentAssets | Net Current Assets | PIT | durchgereicht | Quartal | letzter Quartalsstichtag | ✓ | niedrig |
| calculate_KGV | KGV (P/E) | R | ohne frequency-Param (aktuell, tw. trailingPE-Fallback) | — | unverändert, kein Param | n. z. | — |
| calculate_peg_ratio | PEG Ratio | R | CAGR-basiert, ohne frequency | — | unverändert | n. z. | — |
| calculate_book_value_per_share | Buchwert je Aktie | PIT | fix quarterly-Bilanz (Model.py:422) | — | unverändert (ist bereits Stichtag) | n. z. | — |
| get_current_tbv_and_price | TBV & Kurs | PIT | ohne frequency | — | unverändert | n. z. | — |
| calculate_avg_annual_profit_growth | Ø Jahreswachstum | Wachstum | fix annual (Model.py:1545) | — | **kein TTM v1** (TTM-vs-TTM-Vorjahr wäre neue Numerik) | ✗ | — |
| calculate_avg_quarterly_profit_growth | Ø Quartalswachstum | Wachstum | fix quarterly (Model.py:1421) | — | **kein TTM v1** | ✗ | — |
| compare_avg_annual/quarterly_growth_to_inflation | Wachstum vs. Inflation | Wachstum | fix | — | **kein TTM v1** | ✗ | — |
| calculate_current_dividend_yield | Dividendenrendite (aktuell) | F | bereits TTM-Dividenden | — | unverändert (de facto TTM) | n. z. | — |
| calculate_historical_dividend_yield_average | Div.-Rendite Ø 10 J | F | Jahresresample (Model.py:321) | — | **kein TTM v1** | ✗ | — |
| analyze_payout_ratio | Ausschüttungsquote | R | annual (Model.py:166) | — | **kein TTM v1** (Semantik unklar) | ✗ | — |
| analyze_dividend_history | Dividendenhistorie | F | resample("YE") (Model.py:245) | — | **kein TTM v1** | ✗ | — |
| calculate_annual_inflation_rate / total_inflation | Inflation (Makro) | Makro | ohne frequency | — | unverändert | n. z. | — |
| calculate_historical_* (19 Reihen, Katalog Z. 281-318) | Historische Zeitreihen | HIST | **immer** quarterly→4Q-Rolling-TTM (Model.py:2160 ff.) | identisch | identische Ausgabe; Cache-Keys `*_ttm_*` wiederverwendet | ✓ | niedrig |
| evaluate_tbv_bandwidth / evaluate_ebit_bandwidth | TBV-/EBIT-Bandbreite | komplex | TTM-basiert, ohne frequency-Param | — | unverändert | n. z. | — |
| calculate_crv / course_target_Price-/EVMultiples | CRV / Kursziele | komplex | auf HIST-TTM-Reihen | — | unverändert | n. z. | — |

*Legende „TTM?": ✓ = frequency="ttm" wird akzeptiert (Delegation) · ✗ = Capability-Map schließt ttm aus, klare Fehlermeldung + UI-Ausblendung · n. z. = Metrik hat keinen frequency-Parameter, nicht betroffen.*

**Gemeinsame Angaben je Gruppe:** Datenquelle = SEC companyfacts primär, Alpha Vantage sekundär, yfinance für Kurse (§3). Benötigte Rohdaten für ✓-Metriken = letzte 4 vollständige Quartale (Flow) bzw. letzter Quartalsstichtag (PIT). Fallback = Fehlermeldung „TTM nicht verfügbar" (kein stiller Fallback, §11 E1). Edge Cases = <4 Quartale, fehlendes/dupliziertes Quartal, Foreign Issuer/20-F (kein quarterly → kein ttm), Restatements (Anbieterstand wird übernommen, wie heute), Nullwerte/negative Nenner (heutiges Verhalten des Annual-Pfads gilt unverändert). Tests = EV-137 + „ttm==annual"-Assertions (EV-132).

## 10. Vergleich: Quarterly ersetzen oder TTM ergänzen

| Kriterium | Variante A: Quarterly → TTM ersetzen | Variante B: TTM ergänzen |
|---|---|---|
| Migrationsrisiko | hoch: gespeicherte History-Einträge, Custom-Definitionen (frequency im JSON), Compare-localStorage, API-Clients mit `quarterly` | keines — Bestand bleibt gültig |
| Custom Modes | Migration/Mapping nötig | unverändert |
| Gespeicherte Analysen / Reproduzierbarkeit | Rerun mit `quarterly` bräuchte Legacy-Pfad | unverändert |
| API-Verträge | Enum-Wert verschwindet → Breaking | additiv |
| Nutzerverständnis | einfacher (2 Optionen), aber Quartalssicht verschwindet | 3 Optionen, Tooltip nötig |
| Fachlicher Wert | Einzelquartalssicht (Saisonalität, Zykliker-Modi nutzen quarterly!) ginge verloren | Quartalssicht bleibt für Zykliker/Turnarounds |
| Rollback | schwer (Datenmigration zurück) | trivial („ttm" aus ALLOWED entfernen) |
| Aufwand | M–L + Migrationsrisiko | M, rein additiv |

**Entscheidung (Betreiber, 2026-07-16): Variante B — TTM ergänzen; zusätzlich wird TTM Frontend-Default.** Besondere Stützung durch den Befund, dass die Zykliker-/Turnaround-Modi intern echte Quartalsdaten verwenden (`full_analysis.py:43-67`) — Variante A hätte dort fachliche Folgen gehabt.

## 11. Offene Produktentscheidungen

**Bereits entschieden (Betreiber, 2026-07-16):** ① TTM ergänzen (Variante B), ② TTM wird Frontend-Default, ③ Partikelstrahl auf Mobile durch statischen Gradient ersetzt (Desktop unverändert), ④ Landingpage-Hero-Fokus = Nachvollziehbarkeit schärfen.

**Offen — mit Standardannahme geplant (Umsetzung folgt der Standardannahme, sofern der Betreiber nicht widerspricht):**

| ID | Entscheidung | Standardannahme | Alternative | Auswirkung | Aufgaben |
|---|---|---|---|---|---|
| E1 | <4 vollständige Quartale | TTM „nicht verfügbar" mit klarer Meldung; kein Hochrechnen, kein stiller Fallback | Teil-TTM / Extrapolation | Fehlerpfad statt Schätzwert | EV-132, EV-137 |
| E2 | Anbieter-TTM vs. intern | intern berechnen (de facto bereits so; kein Anbieter liefert TTM-Statements) | yfinance-trailing-Felder als Zusatzquelle | keine | EV-132 |
| E3 | Schwellenwerte bei TTM | unverändert (heutige Schwellen sind implizit auf annual≈TTM kalibriert); quarterly-Inkonsistenz nur dokumentieren | Neukalibrierung | keine Codeänderung | EV-132, EV-138 |
| E4 | TTM-Bezeichnung UI (nur DE-Oberfläche vorhanden) | Label „TTM" + Tooltip „TTM (letzte 12 Monate): Summe der letzten vier berichteten Quartale" | Label „Letzte 12 Monate" | nur Copy | EV-136 |
| E5 | TTM in Analysemodi | Modi akzeptieren ttm via Delegation (nötig für TTM-Default auf AnalyzePage); feste frequencies-Listen der Vollanalyse bleiben unverändert | Modi in v1 ausnehmen (kollidiert mit Default-Entscheidung ②) | EV-132/135 |
| E6 | LP-Zielgruppe | private Selbstentscheider / langfristige Einzelaktien-Anleger | semi-professionelle Analysten | Tonalität Copy-Deck | EV-122 |
| E7 | Live-Preis-Kadenz mobil | 20 s beibehalten, Last sinkt durch Dedup (EV-113) | 60 s auf Mobile | ggf. Folge-Task | EV-113 |
| E8 | Track-Endpoint Stufe 2 | nicht bauen; `?src=`-Attribution reicht | EV-125 umsetzen | EV-125 bleibt „Optional" | EV-124/125 |
| E9 | „1 Analyse = wie viele Einheiten?" | vor EV-121c fachlich beim Betreiber klären (aus Code: `units=1` pro Request, aber Vollanalyse = mehrere Requests?) | — | Wording D3 | EV-121 |
| E10 | Screenshots/Nutzerstimmen für LP | keine neuen Assets in diesem Vorhaben; nur Text | echte Produktscreenshots ergänzen | Scope | EV-122/123 |

## 12. Empfohlene Zielarchitektur

1. **TTM als Alias/Delegation:** `frequency="ttm"` nimmt an den Methoden-Eingängen exakt den heutigen Annual-Codepfad (Flow: 4Q-Summe; PIT: letzter Stichtag; HIST: identische Rolling-Ausgabe); Antwort-Dict trägt `"frequency": "ttm"`. Null neue Numerik, Annual byte-identisch (Golden Master). Bestehende `*_ttm_*`- und `*_quarterly`-Cache-Artefakte werden wiederverwendet — kein neuer Cache-Namespace, da ttm nichts Neues berechnet.
2. **Zentrales Modul `agent/frequency.py`:** `ALLOWED`-Tupel, `validate_frequency()` (textidentische Fehler-Dicts), `resolve_fetch_frequency()` (ttm→quarterly-Rohdaten), Capability-Map je Metrik. Die ~30 verstreuten Guards werden verhaltensneutral darauf umgestellt; ein Eintrag aus `ALLOWED` entfernen = Kill-Switch.
3. **Frontend-Performance ohne neue Dependencies:** modul-globaler Promise-Cache `currentUserCache.ts` (TTL ~30 s + Invalidierung); zentraler `priceStore.ts` (Symbol→Listener-Map, Ref-Counting, EIN 20s-Tick) hinter unveränderter `useLivePrice`-Signatur.
4. **Route-Splitting per Seite, Layouts statisch:** `React.lazy` für alle 22 Seiten, Suspense um das Route-Element (Sidebar/Layout bleiben gemountet), `AppTour`/react-joyride lazy analog IntroOverlay, `lazyWithRetry` gegen Stale-Chunk-404 nach Render-Redeploys; `manualChunks` erst nach Build-Analyse.
5. **Landingpage: Wahrheitsschulden vor Stilistik; Tracking über Bestehendes:** `?src=`-Attribution am `user_registered`-Event; kein neuer öffentlicher Endpoint in v1.

## 13. Umsetzungsphasen

| Phase | Inhalt | Aufgaben |
|---|---|---|
| Phase 0 – Bestehenden Zustand absichern | Baseline + Golden Master | EV-100, EV-101 |
| Phase 1 – Performance messen und priorisieren | in EV-100 enthalten (Messprotokoll → §7) | EV-100 |
| Phase 2 – Mobile Quick Wins | Partikel-Gate, Badge-Pulse | EV-110, EV-111 |
| Phase 3 – Rendering und Datenladen | /me-Dedup, Preis-Store, Code-Splitting, Skeletons | EV-112, EV-113, EV-114, EV-115 |
| Phase 4 – Mobile Charts und Animationen | gezielte Memoisierung, Chart-Kosten | EV-116, EV-117 |
| Phase 5 – Landingpage-Positionierung | SEO-Fixes, Diskrepanzen | EV-120, EV-121 |
| Phase 6 – Landingpage-Content-Plan | Copy-Deck, Umsetzung, Attribution | EV-122, EV-123, EV-124, (EV-125) |
| Phase 7 – TTM-Datenmodell | Frequenz-Modul, Guard-Zentralisierung | EV-130 |
| Phase 8 – TTM-Berechnungen | Capability-Map, Agent-Kern | EV-131, EV-132 |
| Phase 9 – TTM-Integration in Analysen | API-Schicht | EV-133 |
| Phase 10 – TTM in Vergleich und Custom Modes | Frontend-Typen/Toggle/Default | EV-134, EV-135 |
| Phase 11 – UI und Nutzererklärung | TTM-Copy | EV-136 |
| Phase 12 – Regression und Verifikation | Edge-Case-Tests, Doku, Gesamtregression | EV-137, EV-138, EV-139 |

Die drei Themen sind weitgehend unabhängig; Performance (Phasen 2-4), Landingpage (5-6) und TTM (7-12) können parallel bearbeitet werden — innerhalb eines Themas gilt die Phasenreihenfolge strikt.

## 14. Detaillierte Aufgaben

### [EV-100] Mobile-Performance-Baseline erheben

**Status:** Offen · **Bereich:** Performance · **Phase:** 0/1 · **Priorität:** Kritisch · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** keine

#### Ziel
Reproduzierbare Vorher-Messung gemäß Messkonzept §7, damit jede Optimierung einen Ausgangswert hat.

#### Aktueller Zustand / Beweis
Keine Messwerte vorhanden; nur Bundle-Größen gemessen (dist: 1,19 MB / 349 KB gzip).

#### Geplante Änderung
Keine Codeänderung. `npm run build` + `vite preview`, Lighthouse Mobile (Moto-G-Preset, Slow 4G) auf den §7-Seiten in den §7-Szenarien; Network-Tab-Zählung `/me` und `/current-price` auf dem Dashboard; JS-Heap nach 2 min. Ergebnistabelle in §7 dieses Dokuments eintragen.

#### Betroffene Dateien
Nur EVOLVING.md (§7).

#### Implementierungsschritte
1. Production-Build lokal starten, Testnutzer mit ≥10 Favoriten vorbereiten.
2. Messmatrix §7 abarbeiten, je Zelle 3 Läufe, Median notieren.
3. Ergebnistabelle + Auffälligkeiten in §7 eintragen.

#### Tests / Akzeptanzkriterien
- [ ] Für `/`, `/login`, `/app/dashboard`, `/app/analyze`, `/app/compare` liegen LCP/INP/CLS/TBT + Request-Zählungen vor (kalt+warm, Free+Pro).
- [ ] `/me`- und `/current-price`-Anzahl pro Dashboard-Load dokumentiert.

#### Rollback
n. z. (keine Änderung).

---

### [EV-101] Golden-Master-Harness für Annual und Quarterly

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** TTM · **Phase:** 0 · **Priorität:** Kritisch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** keine — **Gate für alle TTM-Tasks**

#### Ziel
Beweisbare Unveränderlichkeit von Annual- und Quarterly-Ergebnissen über alle folgenden Änderungen hinweg.

#### Aktueller Zustand / Beweis
Es existieren Einzeltests (`agent/tests/test_ttm_historical_multiples.py`, `test_fundamental_metrics.py`), aber kein flächiger Snapshot über den Katalog.

#### Geplante Änderung
Neues `agent/tests/test_golden_master_frequencies.py`: für Referenzsymbole (Vorschlag §17: AAPL, AMD, KO, SAP — final anhand Cache-Verfügbarkeit prüfen, Begründung dokumentieren) alle Katalog-Metriken über `api/services/metric_catalog.call_metric` mit `frequency=annual` und `quarterly` aufrufen; Ergebnisse als JSON-Snapshots in `agent/tests/golden/` einchecken. Offline gegen den Datei-Cache `cache/` (use_cache erzwingen, Netzwerkzugriffe mocken/verbieten). Preis-/zeitabhängige Felder (current_price-basierte Ratios, Timestamps) entweder Preis mocken oder Feld per Normalisierungsliste vom Vergleich ausnehmen — Liste im Test dokumentieren. Zusätzlich API-Ebene: FastAPI-TestClient-Snapshots je 1 Route aus `analyze.py`, `metric_routes.py`, `custom_analysis.py`, inkl. Fehlerpfad `frequency=invalid` mit **exaktem** Fehlertext.

#### Betroffene Dateien
`agent/tests/test_golden_master_frequencies.py` (neu), `agent/tests/golden/*` (neu), ggf. `api/tests/test_frequency_golden_master.py` (neu).

#### Zu schützende Funktionen
Alle — dieser Task schützt, er ändert nichts Produktives.

#### Implementierungsschritte
1. Cache-Abdeckung der Kandidatensymbole prüfen (`ls cache/ | grep …`); 4 Symbole festlegen (Large-Cap, Zykliker, Dividendenzahler, Foreign Issuer/20-F).
2. Snapshot-Runner schreiben (Katalog-Iteration, Normalisierungsliste für volatile Felder).
3. Snapshots erzeugen, committen; Zweitlauf muss diff-frei sein (Determinismus-Nachweis).
4. API-TestClient-Snapshots inkl. Fehlertext ergänzen.

#### Akzeptanzkriterien
- [ ] `pytest agent/tests/test_golden_master_frequencies.py` grün und deterministisch (2 Läufe identisch).
- [ ] Fehlertext für `frequency=invalid` ist als Snapshot fixiert.
- [ ] Foreign-Issuer-Symbol deckt den Quarterly-Blockade-Pfad (`sec_source.py:98-106`) ab.

#### Rollback
n. z. (nur Tests).

---

### [EV-110] ParticleBeamBackground: Mobile-Gate mit statischem Ersatz

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** Performance · **Phase:** 2 · **Priorität:** Kritisch · **Aufwand:** XS · **Risiko:** Niedrig · **Abhängigkeiten:** EV-100 (Baseline vorher)

#### Ziel
Kein Canvas-rAF-Loop auf Mobilgeräten; Desktop pixelidentisch unverändert (Produktentscheidung ③).

#### Aktueller Zustand / Beweis
`ParticleBeamBackground.tsx:132-153` (rAF-Loop), `:95-97` (bis 520 Partikel), `:20-26` (Mobile-Basis 180, aber Höhen-Skalierung greift); kein isMobile-Gate — nur `prefers-reduced-motion` (`:75,150,156`). Muster für das Gate: `AmbientBackground.tsx:31-32` (`disableContinuousAnimation = prefersReducedMotion || isMobile`, iOS-Safari-Kommentar). 5 Mount-Points: LandingPage:127, DashBoardPage:207, AccountPage:400, BillingPage:94, AuthLayout:25.

#### Geplante Änderung
In `ParticleBeamBackground.tsx`: `useIsMobile()` (aus `hooks/useMediaQuery`); bei `isMobile` kein Canvas mounten, stattdessen statischer, dem Beam optisch angenäherter Gradient-Div (Farben/Positionen aus den vorhandenen Partikel-Farbkonstanten ableiten). Reduced-Motion-Pfad unverändert.

#### Betroffene Dateien
Nur `frontend/src/components/landing/ParticleBeamBackground.tsx` — die 5 Mount-Points bleiben unangetastet.

#### Zu schützende Funktionen
Desktop-Darstellung aller 5 Seiten (Screenshotvergleich), `densityMultiplier`-Prop-Vertrag.

#### Implementierungsschritte
1. `useIsMobile` importieren, Early-Return mit Gradient-Div vor der Canvas-Logik.
2. Gradient visuell an Beam annähern (Dev-Server, Mobile-Emulation).
3. Desktop-Screenshotvergleich aller 5 Seiten.

#### Tests / Nachweis
Automatisiert: bestehende vitest-Suite grün. Manuell: DevTools-Performance-Trace mobil → kein rAF-Ticker von ParticleBeam; Desktop unverändert. Nachweis: Trace-Screenshot + LCP/TBT-Vergleich zu EV-100 auf Dashboard.

#### Akzeptanzkriterien
- [ ] Mobil: kein Canvas-Element von ParticleBeamBackground im DOM, kein rAF-Loop.
- [ ] Desktop: visuell unverändert.

#### Rollback
Einzelner Commit revert.

---

### [EV-111] LivePriceBadge: Infinite-Pulse durch CSS ersetzen, feste Breite

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** Performance · **Phase:** 2 · **Priorität:** Hoch · **Aufwand:** XS · **Risiko:** Niedrig · **Abhängigkeiten:** EV-100

#### Ziel
Kein JS-Animations-Ticker pro Badge (20+ parallele framer-motion-Loops auf dem Dashboard); kein Layout-Sprung „…"→Preis.

#### Aktueller Zustand / Beweis
`LivePriceBadge.tsx:25-29`: framer-motion-Animation mit `repeat: Infinity` pro Instanz; `:31`: Platzhalter „…" ohne feste Breite.

#### Geplante Änderung
Pulse durch reine CSS-Keyframe-Animation ersetzen (läuft auf Compositor statt JS-Ticker) und `prefers-reduced-motion` respektieren; `min-width` am Badge, damit Preis-Eintreffen das Layout nicht verschiebt.

#### Betroffene Dateien
`frontend/src/components/shared/LivePriceBadge.tsx`.

#### Zu schützende Funktionen
Alle 12 Nutzungsstellen (nur interne Darstellung ändert sich, Props-API bleibt).

#### Implementierungsschritte
1. CSS-Keyframes definieren (Inline-`<style>` oder bestehendes Theme-Muster), motion-Wrapper entfernen.
2. `min-width` festlegen (breitester realistischer Preis-String).
3. Alle Badge-Kontexte visuell prüfen (Sidebar, Dashboard, Analyze, Compare).

#### Akzeptanzkriterien
- [ ] Kein framer-motion-Infinite-Loop mehr in LivePriceBadge (Code-Review + Profiler).
- [ ] Kein sichtbarer Layout-Shift beim Preis-Eintreffen.

#### Rollback
Commit revert.

---

### [EV-112] `/me`-Deduplizierung über `currentUserCache`

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** Performance · **Phase:** 3 · **Priorität:** Hoch · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** EV-100

#### Ziel
Genau 1 `/me`-Request pro Seitenladung statt 4+.

#### Aktueller Zustand / Beweis
Parallele `getCurrentUser()`-Aufrufe: `DashBoardPage.tsx:98`, `AppSidebar.tsx:69`, `EmailVerificationBanner.tsx:18`, `useAppTour.ts:65`, `BillingPage.tsx:40`, `AccountPage.tsx:70`, `AdminRoute.tsx:15`, `useCustomAnalysisDefinitions.ts:32` (+ SupportPage). Kein Cache im fetch-Wrapper.

#### Geplante Änderung
Neues Modul `frontend/src/api/currentUserCache.ts`: `getCurrentUserCached()` teilt In-flight-Promises und cached das Ergebnis ~30 s (modul-global); `invalidateCurrentUser()` exportieren. Invalidierung: nach Profil-Save (AccountPage), nach Billing-Success, und via Listener auf das bestehende `app:logout`-Window-Event (siehe `api/client.ts`). Alle Call-Sites mechanisch umstellen — vorher per `grep -rn "getCurrentUser(" frontend/src` die vollständige Liste verifizieren.

#### Betroffene Dateien
`frontend/src/api/currentUserCache.ts` (neu) + alle grep-verifizierten Call-Sites (~9 Dateien).

#### Zu schützende Funktionen
Anzeige-Aktualität nach Profil-/Plan-Änderung (deshalb kurze TTL + explizite Invalidierung); AdminRoute-Gate darf nie veraltete Rolle verwenden → AdminRoute erhält `forceFresh`-Option oder bleibt auf ungecachtem Aufruf (im Task entscheiden und begründen).

#### Implementierungsschritte
1. Modul schreiben (Promise-Sharing, TTL, Invalidierung, `app:logout`-Listener).
2. Call-Sites umstellen (AdminRoute-Sonderfall dokumentieren).
3. Mutationspfade (AccountPage-Save, Billing-Success/Cancel-Rückkehr) mit `invalidateCurrentUser()` versehen.

#### Tests / Nachweis
Manuell: Network-Tab Dashboard-Load → genau 1 `/me`; Profil ändern → Anzeige aktualisiert; Logout/Login-Wechsel → kein Stale-User. Nachweis: Network-Screenshot vorher (4+) / nachher (1).

#### Akzeptanzkriterien
- [ ] 1 `/me` pro Dashboard-Load (kalt).
- [ ] Nach Profil-Save zeigt Sidebar/Account den neuen Stand ohne Reload.
- [ ] `npm run build` grün.

#### Rollback
Call-Sites zurück auf `getCurrentUser`, Modul löschen.

---

### [EV-113] Zentraler Live-Preis-Store

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** Performance · **Phase:** 3 · **Priorität:** Hoch · **Aufwand:** M · **Risiko:** Mittel · **Abhängigkeiten:** EV-100; sinnvoll nach EV-111

#### Ziel
Pro eindeutigem Symbol genau 1 Poll/20 s (statt ~2N Badge-Instanzen); neu gemountete Badges zeigen sofort den letzten bekannten Wert.

#### Aktueller Zustand / Beweis
`useLivePrice.ts:10,61`: eigenes 20s-Interval pro Hook-Instanz; `document.hidden`-Pause (`:62`) und Cleanup (`:67-71`) vorhanden. Sidebar (1 Badge/Favorit, `AppSidebar.tsx:319,566`) + DashboardFavoritesSection pollen dieselben Symbole doppelt.

#### Geplante Änderung
Neues Modul `frontend/src/hooks/priceStore.ts` (modul-global, keine Dependency): Map `symbol → {price, error, lastFetched, listeners:Set}` mit Ref-Counting via `subscribe(symbol, listener)`; EIN globaler 20s-Tick fetcht nur Symbole mit aktiven Listenern und nur bei `document.visibilityState === "visible"` (bestehende `getCurrentPrice` aus `api/marketData.ts` wiederverwenden); Sofort-Fetch bei Erst-Subscription, Cache-Hit <20 s liefert synchron. `useLivePrice.ts` intern auf den Store umstellen — **Rückgabe-Signatur `{price, error, isLoading}` bleibt identisch**, sodass `LivePriceBadge` und alle 12 Nutzungsstellen unverändert bleiben.

#### Betroffene Dateien
`frontend/src/hooks/priceStore.ts` (neu), `frontend/src/hooks/useLivePrice.ts`.

#### Zu schützende Funktionen
Alle Badge-Kontexte; Pause bei verstecktem Tab; kein Weiterlaufen nach Unmount (Ref-Count → 0 stoppt Symbol; globaler Timer stoppt bei leerer Map).

#### Implementierungsschritte
1. Store mit Ref-Counting + globalem Tick implementieren.
2. `useLivePrice` auf `useSyncExternalStore` oder subscribe/useState umstellen (Signatur identisch).
3. Doppel-Mount-Test (StrictMode!) und Unmount-Test.

#### Tests / Nachweis
Manuell: Network-Tab Dashboard mit N Favoriten → pro 20 s exakt N `/current-price`-Requests (Symbole eindeutig); Seitenwechsel → keine weiteren Requests für nicht mehr sichtbare Symbole; Tab in Hintergrund → 0 Requests. Nachweis: Network-Zählung vorher/nachher.

#### Akzeptanzkriterien
- [ ] Requests/20 s = Anzahl eindeutiger sichtbarer Symbole.
- [ ] Kein Timer/Fetch nach Unmount aller Badges (Store-Map leer).
- [ ] Badges zeigen bei Remount sofort den letzten Wert (kein „…"-Flackern).
- [ ] `npm run build` grün.

#### Rollback
`useLivePrice.ts` auf alten Stand, Store-Datei löschen.

---

### [EV-114] Route-basiertes Code-Splitting

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** Performance · **Phase:** 3 · **Priorität:** Kritisch · **Aufwand:** L · **Risiko:** Mittel · **Abhängigkeiten:** EV-100; nach EV-110–EV-113 (saubere Attribution der Messungen)

#### Ziel
Initiales JS auf Mobile drastisch senken: Haupt-Chunk ohne recharts/react-joyride/Seiten-Code; Ziel < ~130 KB gzip Entry (Baseline 349 KB).

#### Aktueller Zustand / Beweis
`App.tsx:3-43`: 22 statische Seiten-Imports; `vite.config.ts:1-13` ohne Build-Konfig; `AppLayout.tsx:13` importiert `AppTour` (react-joyride) statisch; einziges Lazy-Muster existiert bereits: `AppLayout.tsx:20` (`IntroOverlay`).

#### Geplante Änderung
- Alle 22 Seiten in `App.tsx` auf `React.lazy` umstellen; Helper `withSuspense(LazyPage)` mit minimalem zentrierten Fallback; Suspense-Grenze um das Route-`element`, damit Layout/Sidebar beim Chunk-Load gemountet bleiben.
- **Statisch bleiben:** Layouts (`PublicLayout`, `AuthLayout`, `AppLayout`), `ProtectedRoute`/`AdminRoute`, alle Provider, `CookieConsentBanner`.
- `AppTour` in `AppLayout.tsx` lazy laden (analog IntroOverlay-Muster, `<Suspense fallback={null}>`).
- `lazyWithRetry`-Wrapper: bei Chunk-Import-Fehler (Stale-Hash nach Render-Redeploy) einmalig `window.location.reload()` mit sessionStorage-Guard gegen Reload-Schleifen.
- `manualChunks` in `vite.config.ts` **nicht** setzen; erst nach Build-Analyse erwägen, falls z. B. framer-motion in viele Chunks dupliziert wird.

#### Betroffene Dateien
`frontend/src/App.tsx`, `frontend/src/layouts/AppLayout.tsx`, ggf. `frontend/src/utils/lazyWithRetry.ts` (neu).

#### Zu schützende Funktionen
Alle 22 Routen inkl. Redirects (`/` → Dashboard bei Token; `/app/custom-analysis` → Analyze, `App.tsx:26-30`); IntroOverlay-Verhalten; Provider-Reihenfolge; Deep-Links mit Query-Params.

#### Implementierungsschritte
1. `lazyWithRetry` + `withSuspense` bauen.
2. Seiten-Imports mechanisch umstellen (ein Commit).
3. `AppTour` lazy.
4. `npm run build` → dist analysieren (Chunk-Größen, recharts/joyride nicht im Entry — z. B. via `grep -l recharts` auf Chunk-Inhalte oder Rollup-Output lesen).
5. Alle Routen durchklicken (inkl. eingeloggt/ausgeloggt, Redirects, Direktaufruf tiefer URLs).

#### Tests / Nachweis
Automatisiert: vitest grün, `npm run build` grün. Manuell: vollständiger Routen-Smoke-Test. Nachweis: dist-Größenvergleich vorher/nachher + Lighthouse Mobile TBT/LCP-Delta auf `/` und `/app/dashboard`.

#### Akzeptanzkriterien
- [ ] Entry-Chunk < ~130 KB gzip; recharts und react-joyride nicht im Entry-Chunk.
- [ ] Alle Routen funktionieren; kein Layout-Flackern beim Seitenwechsel (Layout bleibt stehen, nur Content-Fallback).
- [ ] Lighthouse-TBT auf Mobile messbar gesunken (Wert dokumentieren).

#### Rollback
Ein Commit revert (Imports zurück auf statisch).

---

### [EV-115] Skeleton-Strategie gegen zeitversetztes Erscheinen

**Status:** Offen · **Bereich:** Performance · **Phase:** 3 · **Priorität:** Mittel · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** EV-112, EV-113 (weniger Ladephasen zuerst)

#### Ziel
CLS < 0,1 auf dem Dashboard; wahrgenommene Ladezeit senken durch reservierte Flächen statt einpoppender Karten.

#### Aktueller Zustand / Beweis
Gestaffelte isLoading-Zustände ohne Platzhalter: `DashBoardPage.tsx:98 ff.`, `DashboardFavoritesSection.tsx:29-57`; Badges „…"→Wert (EV-111 fixt die Badge-Breite bereits).

#### Geplante Änderung
Wiederverwendbare `Skeleton`-Komponente (CSS-Shimmer, respektiert `prefers-reduced-motion`); Einsatz: Dashboard-Karten mit festen Höhen, Favoriten-Sektion mit Platzhalterzeilen in Favoritenanzahl, Historien-Karte. Kein „App-weiter Blocker" — bereichsweise Skeletons, Above-the-Fold zuerst.

#### Betroffene Dateien
`frontend/src/components/ui/Skeleton.tsx` (neu), `DashBoardPage.tsx`, `DashboardFavoritesSection.tsx`.

#### Akzeptanzkriterien
- [ ] Lighthouse CLS Dashboard < 0,1.
- [ ] Slow-4G-Ladevorgang zeigt reservierte Flächen statt springender Karten (visuelle Prüfung).

#### Rollback
Skeleton-Einsatzstellen revert; Komponente ist additiv.

---

### [EV-116] Gezielte Memoisierung (Profiler-belegt)

**Status:** Offen · **Bereich:** Performance · **Phase:** 4 · **Priorität:** Mittel · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** EV-113 (Store reduziert Re-Renders bereits), EV-114

#### Ziel
Unnötige Re-Renders teurer Komponenten eliminieren — nur dort, wo der React-DevTools-Profiler sie nachweist.

#### Aktueller Zustand / Beweis
0 Treffer für `React.memo` projektweit; `TimeSeriesChart.tsx:50` transformiert Daten bei jedem Render ohne `useMemo`.

#### Geplante Änderung
Kandidaten (jeweils erst Profiler-Nachweis, dann Umsetzung, dann Gegenmessung): `TimeSeriesChart` (Transformation in `useMemo`, Komponente in `React.memo` — Props-Stabilität vorher prüfen), `LivePriceBadge` (`React.memo`), Sidebar-Favoritenzeile. Ausdrücklich **nicht** flächendeckend; Kandidaten ohne Profiler-Beleg werden verworfen und hier dokumentiert.

#### Betroffene Dateien
`frontend/src/components/charts/TimeSeriesChart.tsx`, `LivePriceBadge.tsx`, `AppSidebar.tsx`.

#### Akzeptanzkriterien
- [ ] Profiler-Vorher/Nachher je Kandidat dokumentiert.
- [ ] Keine visuellen/funktionalen Regressionen in Charts (Tooltip, Resize).

#### Rollback
Pro Kandidat einzeln revertierbar.

---

### [EV-117] Performance-Nachmessung und Regressionsprüfung

**Status:** Offen · **Bereich:** Performance · **Phase:** 4 · **Priorität:** Hoch · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** EV-110–EV-116

#### Ziel
Wirkung belegen, Desktop-Regressionen ausschließen.

#### Geplante Änderung
EV-100-Messmatrix identisch wiederholen; Vorher/Nachher-Tabelle in §7 ergänzen; zusätzlich Desktop-Lighthouse auf `/` und `/app/dashboard` (darf sich nicht verschlechtern); Langzeittest: 10 min Navigation ohne Reload → JS-Heap stabil (kein Wachstum durch priceStore/Skeletons).

#### Akzeptanzkriterien
- [ ] Mobile TBT und LCP auf Dashboard und Landing signifikant gesunken (Zielwert: TBT −50 % gegenüber Baseline; falls verfehlt, Abweichung begründen und nächste Maßnahme ableiten).
- [ ] Desktop-Werte nicht verschlechtert; Heap stabil.

---

### [EV-120] SEO-Grundfixes in `frontend/index.html`

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** Landingpage · **Phase:** 5 · **Priorität:** Hoch · **Aufwand:** XS · **Risiko:** Niedrig · **Abhängigkeiten:** keine

#### Ziel
Korrekte Sprach-/Meta-Auszeichnung und teilbare Vorschau.

#### Aktueller Zustand / Beweis
`index.html:2` `lang="en"` bei deutschem Inhalt; Title statisch „ComAnalysis"; keine meta description, keine OG-Tags; Favicon `/vite.svg`.

#### Geplante Änderung
`lang="de"`; Title „ComAnalysis — Fundamentalanalyse mit nachvollziehbaren SEC-Daten" (o. ä., final im Copy-Deck EV-122); `meta name="description"` (~150 Zeichen, neutral, keine Anlageberatungs-Konnotation); OG-Tags (og:title, og:description, og:type=website, og:url; og:image nur falls Asset existiert — sonst weglassen, E10); eigenes einfaches SVG-Favicon statt vite.svg.

#### Akzeptanzkriterien
- [ ] Lighthouse-SEO ≥ 90; HTML valide; Social-Preview zeigt Title+Description.

#### Rollback
Trivial (eine Datei).

---

### [EV-121] Landingpage-Diskrepanzen beheben (Wahrheitsschulden)

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** Landingpage · **Phase:** 5 · **Priorität:** Kritisch · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** E9 (Einheiten-Klärung) für Teil c

#### Ziel
Kein Claim auf der Landingpage, den das Produkt nicht deckt (D1–D3 aus §8).

#### Geplante Änderung
a) Checklisten-Punkt und Datenbasis-Strip präzisieren: „Fundamentaldaten direkt aus SEC-Filings. Kursdaten von etablierten Marktdatenanbietern." (Formulierung final im Copy-Deck).
b) Kursaktualitäts-Hinweis ergänzen (Fußnote/Kleintext): „Kurse ca. 15 Minuten verzögert" — an jeder Stelle, die Kurse erwähnt; App-interne Badge-Beschriftung ist **nicht** Scope dieses Tasks (Follow-up notieren).
c) „50 Analysen/Monat" ↔ „Analyse-Einheiten": nach Klärung E9 einheitliche Terminologie über `LandingPage.tsx`, `PricingPage.tsx`, `pricingPlans.ts`-Texte und BillingPage — in einem Commit.

#### Betroffene Dateien
`frontend/src/pages/public/LandingPage.tsx`, `frontend/src/pages/public/PricingPage.tsx`, `frontend/src/config/pricingPlans.ts`, ggf. `BillingPage.tsx` (nur Strings).

#### Zu schützende Funktionen
Keine Struktur-/Logikänderung; Neutralitäts-Disclaimer bleibt.

#### Akzeptanzkriterien
- [ ] D1–D3 behoben; Claim-Beleg-Checkliste (jede Aussage → Produktbeleg) im PR dokumentiert.
- [ ] Terminologie identisch auf Landing/Pricing/Billing.

#### Rollback
String-Reverts.

---

### [EV-122] Copy-Deck für alle Landingpage-Sektionen

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** Landingpage · **Phase:** 6 · **Priorität:** Hoch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** EV-121 (Fakten fixiert), E6

#### Ziel
Vollständige finale DE-Texte je Sektion, vom Betreiber freigegeben, **bevor** Strings ersetzt werden.

#### Geplante Änderung
Copy-Deck als neuer Abschnitt in dieser EVOLVING.md (unter §20 anhängen): je Sektion Ziel, primäre Botschaft, Text neu (final), primärer/sekundärer CTA, erwartete Nutzerfrage, Vertrauensbeleg. Richtung gemäß §8: Hero = Nachvollziehbarkeit geschärft (Produktentscheidung ④), Value-Cards = 8 Modi / Custom-Builder / Vergleich+historische Charts, How-it-works = realer Flow, Stat-Readout „Account & Billing" ersetzen, Final-CTA konkretisieren. Stilregeln: klar, konkret, belegbar, keine Buzzwords/Superlative/KI-Marketingfloskeln; verbotene Muster („Revolutioniere…", „Maximiere Gewinne", Garantie-Aussagen).

#### Akzeptanzkriterien
- [ ] Copy-Deck vollständig (alle Sektionen inkl. CTAs) und vom Betreiber freigegeben.
- [ ] Jeder Satz hat einen Beleg-Vermerk (Produktfunktion/Datenquelle).

#### Rollback
n. z. (Dokument).

---

### [EV-123] Landingpage-Sektionstexte umsetzen

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** Landingpage · **Phase:** 6 · **Priorität:** Hoch · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** EV-122 (freigegebenes Copy-Deck)

#### Geplante Änderung
Ausschließlich String-Ersetzungen in `LandingPage.tsx` (und ggf. `Header.tsx`/`Footer.tsx`-Labels) gemäß Copy-Deck. **Kein** Layout-, Struktur- oder Komponentenumbau; Sektionsreihenfolge bleibt.

#### Zu schützende Funktionen
Responsive Verhalten (svh-Hero, gestapelte Duo-Karten), Auto-Rotation, reduced-motion-Guards, CTA-Routing, SEO-Headings (genau ein h1).

#### Akzeptanzkriterien
- [ ] Texte = Copy-Deck 1:1; Desktop/Tablet/Mobile visuell geprüft; keine Überläufe.
- [ ] h1/h2-Hierarchie unverändert korrekt; `npm run build` grün.

#### Rollback
String-Reverts.

---

### [EV-124] CTA-Attribution über `?src=` am Registrierungs-Event

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** Landingpage · **Phase:** 6 · **Priorität:** Mittel · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** keine (parallel zu EV-122/123 möglich)

#### Ziel
Messbar machen, welcher Landingpage-CTA Registrierungen bringt — ohne neue Tracking-Infrastruktur, DSGVO-neutral (kein zusätzliches Client-Tracking, nur ein Query-Param im eigenen Funnel).

#### Aktueller Zustand / Beweis
Kein CTA-Event; Backend loggt `user_registered` via `api/services/event_service.log_event` → `product_events` (Admin-Dashboard) ohne Quelle.

#### Geplante Änderung
Landingpage-CTAs verlinken auf `/register?src=hero|value|final|pricing|header`; RegisterPage liest `src` (Whitelist!) und übergibt es beim Register-Call; Registrierungs-Route hängt es als Metadata an das bestehende `user_registered`-Event. Kein neuer Endpoint, kein Cookie.

#### Betroffene Dateien
`LandingPage.tsx`, `Header.tsx` (CTA-Links), `frontend/src/pages/public/RegisterPage.tsx` (o. ä. Pfad, per grep verifizieren), `frontend/src/api/auth.ts`, `api/routes/…` (Register-Route), `api/services/event_service.py`-Aufrufstelle.

#### Zu schützende Funktionen
Registrierung ohne `src` (Direktaufruf) funktioniert unverändert; Backend validiert `src` gegen Allowlist (kein freies Nutzer-Input in Events).

#### Akzeptanzkriterien
- [ ] Registrierung über jeden CTA erzeugt `user_registered` mit korrektem `src` in `product_events`.
- [ ] Registrierung ohne `src` unverändert; ungültiges `src` wird verworfen.

#### Rollback
Param ignorieren (rein additiv).

---

### [EV-125] (Optional) Öffentlicher Track-Endpoint für CTA-Klicks

**Status:** Offen (zurückgestellt, E8) · **Bereich:** Landingpage · **Phase:** 6 · **Priorität:** Niedrig · **Aufwand:** M · **Risiko:** Mittel (Spam/Datenschutz) · **Abhängigkeiten:** EV-124 zeigt Bedarf

Nur umsetzen, falls Klickzahlen nicht-konvertierender Besucher benötigt werden: `POST /events/track` mit striktem Event-Namen-Enum (`landing_cta_click` …), IP-Rate-Limit, fire-and-forget, Frontend feuert nur bei erteiltem Analytics-Consent (bestehendes Consent-Gate). Standardannahme: **nicht bauen.**

---

### [EV-130] Zentrales Frequenz-Modul + Guard-Zentralisierung (verhaltensneutral)

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** TTM · **Phase:** 7 · **Priorität:** Kritisch · **Aufwand:** M · **Risiko:** Niedrig (testgesichert) · **Abhängigkeiten:** EV-101 (grün)

#### Ziel
Eine einzige Quelle der Wahrheit für erlaubte Frequenzen; Vorbereitung für „ttm" ohne jede Verhaltensänderung.

#### Aktueller Zustand / Beweis
~30 verstreute Guards `if frequency not in ["annual","quarterly"]`: `DataLoader.py` ×13 (u. a. :168,295,479,1340,1633), `Model.py` ×9 (u. a. :824,904,1073,1146,1833), `sec_source.py` ×6 (:63,182,324,439,581), `AgentAction.py:533`; Enum-Quelle `metric_catalog.py:68-71`.

#### Geplante Änderung
Neues `agent/frequency.py`: `ALLOWED: tuple = ("annual", "quarterly")` (**noch ohne ttm**), `validate_frequency(freq, allowed=ALLOWED)` gibt bei Verstoß exakt die heutigen Fehler-Dicts zurück (Fehlertexte je Aufrufstelle prüfen — falls sie variieren, nimmt `validate_frequency` den Text als Parameter, damit jede Stelle textidentisch bleibt). Alle Guards mechanisch ersetzen; `metric_catalog.py` bezieht `FREQUENCY_PARAM.enum_values` aus dem Modul.

#### Zu schützende Funktionen
Alle — dieser Schritt ist per Definition verhaltensneutral; EV-101-Snapshots (inkl. Fehlertext-Snapshot) sind der Beweis.

#### Implementierungsschritte
1. Alle Guard-Stellen per `grep -rn 'not in \["annual"' agent/ api/` erfassen und ihre Fehlertexte katalogisieren.
2. Modul schreiben; Guards in einem Commit ersetzen.
3. grep-Nachweis: 0 Rest-Treffer; EV-101 laufen lassen.

#### Akzeptanzkriterien
- [ ] Golden Master (EV-101) grün und diff-frei.
- [ ] `grep -rn 'not in \["annual"' agent/ api/` → 0 Treffer.
- [ ] `pytest agent/tests api/tests` grün.

#### Rollback
Commit revert.

---

### [EV-131] Capability-Map je Metrik (TTM-Matrix im Code verankern)

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** TTM · **Phase:** 8 · **Priorität:** Hoch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** EV-130

#### Ziel
Maschinell auswertbare Festlegung, welche Metrik welche Frequenzen unterstützt (Grundlage für Backend-Validierung und UI-Ausblendung).

#### Geplante Änderung
Capability-Angabe je `MetricSpec` in `metric_catalog.py` (bzw. im Frequenz-Modul): Default `["annual","quarterly","ttm"]` für Metriken mit `FREQUENCY_PARAM`; Ausschlüsse gemäß §9 (Dividenden-Metriken, AAGR/AQGR, Payout, Wachstum-vs-Inflation: ohne ttm — sie haben ohnehin keinen frequency-Param, die Capability dokumentiert das explizit). `to_catalog_entry()` gibt Capabilities aus (für EV-134). **Vor der Umsetzung:** ROIC-Nenner-Definition in `Model.py:5291 ff.` lesen und in §9 präzisieren (Ø-Kapital vs. Stichtag) — Matrix-Zeile aktualisieren.

#### Akzeptanzkriterien
- [ ] Jede Katalog-Metrik hat explizite Capabilities; §9-Matrix und Code stimmen überein (Review-Abgleich).
- [ ] Katalog-API-Response enthält Capabilities; EV-101 weiter grün (Response-Erweiterung ist additiv — Golden-Master-Normalisierung ggf. um das neue Feld ergänzen und das dokumentieren).

#### Rollback
Capabilities-Feld entfernen.

---

### [EV-132] `ttm` im Agent-Kern (Delegation)

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** TTM · **Phase:** 8 · **Priorität:** Kritisch · **Aufwand:** L · **Risiko:** Mittel (durch Golden Master gedeckelt) · **Abhängigkeiten:** EV-130, EV-131

#### Ziel
`frequency="ttm"` funktioniert für alle ✓-Metriken aus §9 und liefert exakt die Werte des heutigen Annual-Pfads; Annual/Quarterly byte-identisch.

#### Geplante Änderung
- `ALLOWED` um `"ttm"` erweitern (ein zentraler Ort, EV-130 sei Dank).
- An den Methoden-Eingängen in `Model.py` (frequency-Param-Methoden aus §9) und `AgentAction.py`: `ttm` nimmt denselben Codepfad wie heutiges `annual` (bei KUV/ROE etc. wörtlich derselbe Branch; keinerlei Änderung **im** Annual-Branch); Antwort-Dicts tragen `"frequency": "ttm"` zurück, wo das Feld existiert.
- Rohdaten-Beschaffung: `resolve_fetch_frequency("ttm")` → quarterly (wie der Annual-TTM-Pfad heute schon lädt); bestehende Cache-Keys (`*_quarterly`, `*_ttm_*`) werden wiederverwendet — es entstehen keine neuen Cache-Namespaces, weil nichts Neues berechnet wird.
- Fehlerpfade: <4 Quartale → „TTM nicht verfügbar"-Fehler (E1, exakt definierter Text); Foreign Issuer/20-F → gleiche saubere Fehlermeldung wie heute bei quarterly (`sec_source.py:98-106`-Pfad greift automatisch, verifizieren).
- Analysemodi (`AgentAction.py`-Einstiege): akzeptieren ttm via Delegation (E5); die festen frequencies-Listen der Vollanalyse (`full_analysis.py:43-67`, `AgentOrchestrator.py:9-15`) bleiben **unverändert**.

#### Zu schützende Funktionen
Annual-Formeln/-Ergebnisse (Golden Master), Quarterly-Ergebnisse, Cache-Bestand, intern hartkodierte Frequenzen (`Model.py:422,1421,1545`, historische Reihen) bleiben unangetastet.

#### Implementierungsschritte
1. Für jede ✓-Metrik aus §9: Eingang um ttm-Zweig erweitern (Delegation), Antwort-Feld setzen.
2. Neue Test-Assertions „ttm-Ergebnis == annual-Ergebnis" für alle ✓-Metriken (macht die Aliasbeziehung zum expliziten Vertrag).
3. Fehlerpfad-Tests: Dividenden-Metrik + ttm → Capability-Fehler; Foreign Issuer + ttm; <4 Quartale (Fixture mit gekürztem Cache).
4. EV-101 laufen lassen (muss diff-frei grün sein).

#### Akzeptanzkriterien
- [ ] `ttm == annual`-Assertions grün für alle ✓-Metriken (4 Referenzsymbole).
- [ ] Golden Master diff-frei; Fehlerpfade mit exakten Texten getestet.
- [ ] Kein neuer Cache-Key-Typ entstanden (Verifikation via Cache-Verzeichnis-Diff im Test).

#### Rollback
„ttm" aus `ALLOWED` entfernen → Feature vollständig deaktiviert (Kill-Switch); Delegations-Zweige sind dann toter, harmloser Code und können nachgelagert entfernt werden.

---

### [EV-133] API-Schicht für `ttm`

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** TTM · **Phase:** 9 · **Priorität:** Hoch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** EV-132

#### Geplante Änderung
- Routen (`analyze.py`, `metric_routes.py`, `custom_analysis.py`) validieren über das Frequenz-Modul; Query-Param bleibt `str`, **Default bleibt `"annual"`** (API-Vertrag).
- `FREQUENCY_PARAM.enum_values` enthält „ttm" (kommt via EV-130/132 automatisch aus `ALLOWED` — verifizieren); Katalog-Response mit Capabilities (EV-131).
- grep-Verifikation in `api/` und `frontend/src`, dass kein Read-/Anzeige-Pfad eine Zwei-Werte-Annahme hartcodiert (z. B. `frequency === "quarterly" ? … : …`-Ternaries wie `ComparePage.tsx:50-53` — Liste erstellen und je Stelle prüfen, ob ttm im Else-Zweig korrekt als Jahres-Bucket behandelt wird).
- `analysis_history.frequency` (String(20)) nimmt „ttm" ohne Migration auf; History-Rerun mit ttm testen (`AnalyzePage.tsx:349,367-370`-Pfad, serverseitig `analyze.py:271` Result-Key `"<NAME>|ttm"`).

#### Akzeptanzkriterien
- [ ] TestClient: ttm-Happy-Path je Route; ttm auf Dividenden-Metrik → Capability-Fehler; `frequency=invalid` → alter Fehlertext (Snapshot).
- [ ] History-Eintrag mit `frequency="ttm"` wird gespeichert und rerun-fähig.
- [ ] Golden Master weiter grün.

#### Rollback
Wie EV-132 (Kill-Switch), plus Route-Diff revert.

---

### [EV-134] Frontend: Typen, FrequencyToggle, Capability-Ausblendung

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** TTM · **Phase:** 10 · **Priorität:** Hoch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** EV-133

#### Geplante Änderung
- Zentraler Typ `Frequency = "annual" | "quarterly" | "ttm"` (neu in `frontend/src/types/`), importiert von `types/customAnalysis.ts:1`, `hooks/compareContextValue.ts:4`, `FrequencyToggle.tsx`, `api/analysis.ts:100`, `api/customAnalysis.ts:23`, `api/dataSource.ts:12`, `SourceBadge.tsx`.
- `FrequencyToggle.tsx`: Option `{value: "ttm", label: "TTM"}`; neues optionales Prop `availableFrequencies?: Frequency[]` (Default alle drei) — Kontexte ohne TTM-Support (z. B. Dividenden-Metrik im Custom-Builder) speisen es aus den Katalog-Capabilities (EV-131) und blenden ttm aus.
- Defensive Persistenz: Compare-Load aus localStorage `compare_workspace_v2` validiert frequency gegen Whitelist mit Fallback `"annual"` — schützt beim Rollback (neuer Client schreibt „ttm", alter Code liest es → darf nicht ungefiltert ans Backend gehen).
- `ComparePage.tsx:50-53` `bucketMode`: ttm → `"year"`-Bucket (wie annual), explizit prüfen.

#### Akzeptanzkriterien
- [ ] Toggle zeigt TTM in Analyze/Compare/Custom; Ausblendung greift bei Metriken ohne Capability.
- [ ] Request-Payload `frequency=ttm` korrekt; localStorage-Fallback-Test (manipulierter Wert → „annual").
- [ ] `npm run build` grün (echtes Build).

#### Rollback
Typ-Union und Toggle-Option zurücknehmen; localStorage-Whitelist bleibt (harmlos, schützt weiterhin).

---

### [EV-135] TTM als Frontend-Default

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** TTM · **Phase:** 10 · **Priorität:** Mittel · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** EV-134

#### Ziel
Produktentscheidung ② umsetzen: Neue Auswahl-Defaults sind `ttm` — ohne Ergebnisänderung, da ttm numerisch dem heutigen annual entspricht (nur das Label ändert sich).

#### Geplante Änderung
`AnalyzePage.tsx:112` Initial-State `"ttm"`; `useCompare.tsx:31,36,40` Default für **neuen** State `"ttm"` (vorhandene localStorage-Werte werden respektiert); Custom-Builder-Default je Metrik `"ttm"`, wo Capability es erlaubt, sonst `"annual"`. **Backend-Query-Defaults bleiben `"annual"`** (EV-133). History-Rerun stellt weiterhin die gespeicherte Frequenz wieder her.

#### Zu schützende Funktionen
Gespeicherte Custom-Definitionen (bestehende `"annual"`/`"quarterly"`-Werte im JSON bleiben unangetastet); Rerun-Pfad; `SourceBadge.tsx`-Default prüfen (Anzeige-only).

#### Akzeptanzkriterien
- [ ] Frischer Browser (leerer localStorage): Analyze/Compare/Custom starten mit TTM.
- [ ] Bestehende localStorage-/DB-Zustände unverändert wirksam.

#### Rollback
Default-Strings zurück auf „annual" (3 Stellen).

---

### [EV-136] TTM-UI-Copy und Nutzererklärung

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** TTM · **Phase:** 11 · **Priorität:** Mittel · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** EV-134

#### Geplante Änderung
Tooltip/Hilfetext am Toggle: „TTM (letzte 12 Monate): Summe der letzten vier berichteten Quartale" (E4); Ergebnisüberschriften/History-Anzeige zeigen „TTM" (`AnalysisFrequencyPanel.tsx:23` rendert `toUpperCase()` — „TTM" erscheint automatisch korrekt, verifizieren); Chart-/Vergleichs-Beschriftungen prüfen; Fehlermeldungstext für nicht verfügbare Metriken nutzerverständlich („TTM ist für diese Kennzahl nicht verfügbar — bitte Annual wählen"). Hinweis-Formulierung, dass TTM und Annual derzeit für viele Kennzahlen identische Werte zeigen, im Tooltip **nicht** verschweigen — Formulierung mit Betreiber abstimmen (Teil des Copy-Reviews).

#### Akzeptanzkriterien
- [ ] Tooltip auf Desktop (hover) und Mobile (tap) erreichbar.
- [ ] Alle TTM-Beschriftungen konsistent; Fehlertext verständlich.

---

### [EV-137] TTM-Berechnungs- und Edge-Case-Tests

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** TTM · **Phase:** 12 · **Priorität:** Hoch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** EV-132

#### Geplante Änderung
Pytest-Suite mit synthetischen/gekürzten Cache-Fixtures: 4 vollständige Quartale (Happy Path), ≥5 Quartale (nur jüngste 4 zählen), <4 Quartale (E1-Fehler), fehlendes Quartal in der Mitte, dupliziertes Quartal, Foreign Issuer, Nullwerte, negative Werte (negativer TTM-Nenner bei Multiples → heutiges Annual-Verhalten, dokumentieren), Unternehmen mit kurzer Historie. Getrennt je Kategorie (Flow/PIT/Margen/Multiples). **Manueller Plausibilitätsnachweis** (in Test-Docstring dokumentiert): TTM-Revenue eines Referenzsymbols = Q1+Q2+Q3+Q4 aus `cache/<SYM>_historical_income_statement_quarterly.json` nachgerechnet, gegen API-Response und Frontend-Anzeige verglichen.

#### Akzeptanzkriterien
- [ ] Alle Edge-Case-Tests grün; Plausibilitätsrechnung dokumentiert.
- [ ] Kein stiller Fallback: <4 Quartale liefert nachweislich den definierten Fehler.

---

### [EV-138] Dokumentation der Perioden-Semantik + Follow-up

**Status:** Erledigt (2026-07-16, Sonnet 5) · **Bereich:** TTM · **Phase:** 12 · **Priorität:** Mittel · **Aufwand:** XS · **Risiko:** Keins · **Abhängigkeiten:** EV-132

#### Geplante Änderung
Abschnitt in dieser EVOLVING.md (unter §22 anhängen) + Kommentar an zentraler Stelle (`agent/frequency.py`-Docstring): „`annual` hat seit 2026-07-11 für aktuelle Flow-Kennzahlen und alle historischen Reihen TTM-Semantik; `ttm` ist der explizite Name dieses Pfads; Dividenden/AAGR/Payout nutzen echte Jahresdaten." Follow-up-Ticket notieren (separates, **nicht** begonnenes Vorhaben): „annual → echte Geschäftsjahresdaten umstellen" — die TTM-Default-Entscheidung ② schützt Nutzer vor diesem späteren Semantikwechsel, weil die meisten dann bereits auf ttm arbeiten. Ebenso dokumentieren: bestehende Quarterly-Schwellen-Inkonsistenz (E3).

#### Ergebnis (2026-07-16)
Dokumentiert in `agent/frequency.py` (Moduldocstring + Kommentare bei `ALLOWED_FREQUENCIES`, `MODE_ALLOWED_FREQUENCIES`, `resolve_ttm_alias`, `TTM_CAPABLE_METRICS`) und hier in §22 (siehe unten). Zusätzlicher, während der Umsetzung entdeckter und dokumentierter Befund: die 5 "Whole-Mode"-AgentAction-Methoden reichen `frequency` größtenteils unverändert an bereits TTM-fähige Model.py-Methoden durch — nur `analyze_wachstumswerte` hatte eine eigene riskante `if frequency == "annual"`-Verzweigung (AAGR-vs-AQGR-Wahl), die deshalb explizit mit `resolve_ttm_alias` abgesichert wurde (siehe `agent/tests/test_ttm_mode_methods_ev135.py`). Dieser Fund war nicht in der ursprünglichen Kennzahlenmatrix (§9) erfasst und wird hiermit nachgetragen.

**Follow-up-Ticket (separates, nicht begonnenes Vorhaben):** „annual → echte Geschäftsjahresdaten umstellen" — `annual` berechnet für die 15 TTM_CAPABLE_METRICS und alle `calculate_historical_*`-Reihen weiterhin TTM (Ist-Zustand seit 2026-07-11, durch diese Umsetzung nicht verändert). Ein späterer Wechsel von `annual` zu echten Jahresabschlusswerten würde die Annual-Golden-Master-Snapshots (`agent/tests/golden/`) bewusst brechen (erwarteter Diff, im PR zu begründen) und ist durch die TTM-Default-Entscheidung entschärft, da Nutzer ab sofort überwiegend `ttm` verwenden.

**Bestehende, unveränderte Inkonsistenz (E3, nur dokumentiert):** Quarterly-Einzelwerte werden weiterhin gegen dieselben (auf TTM/Jahresbasis kalibrierten) Schwellenwerte geprüft wie Annual/TTM in den Analysemodi — keine Neukalibrierung in diesem Vorhaben.

---

### [EV-139] Gesamtregression und Abschlussverifikation

**Status:** Teilweise erledigt (2026-07-16, Sonnet 5) · **Bereich:** Übergreifend · **Phase:** 12 · **Priorität:** Kritisch · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** alle vorherigen

#### Geplante Änderung
Keine — reiner Verifikationslauf: Golden Master (EV-101) final; komplette pytest-Suite (`agent/tests`, `api/tests`); vitest; `npm run build`; manueller Durchlauf aller Frontend-Kontexte (Analyze mit allen 3 Frequenzen, Custom-Definition erstellen/laden mit quarterly-Bestand, Compare mit localStorage-Altzustand, History-Rerun annual+quarterly+ttm, Dashboard/Favoriten/Preise, Billing-Smoke); Lighthouse-Abschlussmessung (EV-117-Werte bestätigen); §24-Checkliste abhaken.

#### Ergebnis (2026-07-16)
Automatisiert vollständig ausgeführt und grün: Golden Master (3× diff-frei), `pytest agent/tests api/tests` (328 grün), `npx vitest run` (70 grün), echter `npm run build` (Entry-Chunk 138 KB gzip). **Nicht** ausgeführt: der manuelle, authentifizierte Browser-Durchlauf aller Frontend-Kontexte und die Lighthouse-Abschlussmessung — beide benötigen einen laufenden Test-Account/Postgres bzw. echte Geräte-/Lighthouse-Messung, die in dieser Sitzung nicht verfügbar waren (siehe §24, offene Punkte). Diese verbleiben als offene Nacharbeit vor einem Produktions-Release.

#### Akzeptanzkriterien
- [x] Automatisierte Punkte aus §24 abgehakt und in EVOLVING.md protokolliert.
- [ ] Manuelle/Lighthouse-Punkte aus §24 (benötigen Live-Umgebung) — offen.

## 15. Abhängigkeitsmatrix

```text
EV-100 ──► EV-110, EV-111, EV-112, EV-113, EV-114 (Baseline vor Optimierung)
EV-111 ──► EV-113 (Badge zuerst entschlacken)
EV-112, EV-113 ──► EV-115 ──► EV-116 ──► EV-117
EV-110…116 ──► EV-117

EV-121 ──► EV-122 ──► EV-123
E9 (Klärung) ──► EV-121c
EV-120, EV-124 unabhängig; EV-125 optional nach EV-124

EV-101 ──► EV-130 ──► EV-131 ──► EV-132 ──► EV-133 ──► EV-134 ──► EV-135, EV-136
EV-132 ──► EV-137, EV-138
alle ──► EV-139
```

Themenübergreifend keine Abhängigkeiten — Performance, Landingpage und TTM sind parallel bearbeitbar.

## 16. Performance-Testmatrix

| Dimension | Ausprägungen |
|---|---|
| Geräte | aktuelles iPhone (Safari), älteres iPhone, aktuelles Android (Chrome), leistungsschwaches Android; ersatzweise DevTools-Emulation Moto G + 4× CPU-Throttle |
| Netz | Slow 4G, normales WLAN |
| Cache | kalt, warm |
| Nutzer | ausgeloggt, eingeloggt Free, eingeloggt Pro, ≥10 Favoriten, große Analyse in History |
| Orientierung | Hochformat, Querformat (stichprobenweise) |
| Abläufe | Erstladen je Seite, schneller Seitenwechsel Dashboard↔Analyze, Chart öffnen/schließen ×5, 10 min Nutzung ohne Reload (Heap) |
| Prüfpunkte | Ladezeit (FCP/LCP), Interaktion (INP), Main-Thread (TBT, Long Tasks), CLS, Requests gesamt + `/me`-Zahl + `/current-price`-Zahl, laufende Timer/rAF nach Seitenwechsel, Heap-Entwicklung |

## 17. Annual-Golden-Master-Matrix

Referenzsymbole (Vorschlag; final in EV-101 anhand Cache-Abdeckung, Begründung dort dokumentieren):

| Symbol | Profil | Deckt ab |
|---|---|---|
| AAPL | profitables Large-Cap, Buybacks | Standardpfad, Pro-Aktie-Effekte |
| AMD | Zykliker, saisonale Quartale | Quartals-Oszillation, TTM-Rolling |
| KO | Dividendenzahler | Dividenden-/Payout-Pfade (echte Jahresdaten) |
| SAP (o. ä. 20-F-Filer) | Foreign Issuer | Quarterly-Blockade-Pfad (`sec_source.py:98-106`) |

Vergleichspunkte je Symbol × frequency (annual, quarterly): normalisierte Rohdaten-Auszüge (Stichprobe), alle Katalog-Kennzahlen (call_metric-Ergebnisse), Fehler-Antworten (invalid frequency, Foreign-Issuer-quarterly), API-Responses (TestClient, 3 Routen), gespeicherter History-Snapshot-Aufbau. Frontend-Darstellung wird manuell in EV-139 abgeglichen (kein Snapshot). Volatile Felder (aktueller Kurs, Timestamps) über dokumentierte Normalisierungsliste ausgenommen.

## 18. Quarterly-Regressionsmatrix

Da Variante B (behalten) gewählt wurde: identische Snapshots wie §17 mit `frequency=quarterly` — vor und nach jeder TTM-Phase diff-frei. Zusätzlich manuell in EV-139: Custom-Definition mit gespeicherter quarterly-Metrik lädt und rechnet unverändert; Compare mit quarterly-`bucketMode="quarter"` unverändert; History-Rerun eines alten quarterly-Eintrags liefert plausibel gleiche Struktur; Zykliker-/Turnaround-Modi (nutzen intern quarterly) unverändert.

## 19. TTM-Testmatrix

| Testfall | Erwartung | Task |
|---|---|---|
| 4 vollständige Quartale | ttm == annual (Delegation) für alle ✓-Metriken | EV-132 |
| ≥5 Quartale | nur jüngste 4 zählen | EV-137 |
| <4 Quartale | definierter „nicht verfügbar"-Fehler (E1), kein stiller Fallback | EV-137 |
| fehlendes/dupliziertes Quartal | Fehler bzw. dokumentiertes Verhalten des Annual-Pfads | EV-137 |
| Foreign Issuer (20-F) + ttm | gleiche saubere Fehlermeldung wie quarterly heute | EV-132 |
| Dividenden-Metrik + ttm | Capability-Fehler, UI blendet Option aus | EV-132/134 |
| Nullwerte / negative Nenner | heutiges Annual-Verhalten (dokumentiert) | EV-137 |
| kurze Historie / IPO | <4-Quartale-Pfad | EV-137 |
| History-Persistenz „ttm" | speichern + rerun | EV-133 |
| localStorage-Altzustand / Rollback-Schutz | Whitelist-Fallback „annual" | EV-134 |
| Frequenz-Default frisch | ttm vorausgewählt, Bestand unangetastet | EV-135 |
| Plausibilität | TTM-Revenue = Q1+Q2+Q3+Q4 manuell nachgerechnet (Rohdaten → API → UI) | EV-137 |

Nicht getestet in v1 (bewusst, da kein ttm angeboten): TTM-Wachstumsraten (bräuchten ≥8 Quartale), TTM-Dividendensummen, Währungswechsel/Splits über TTM-Fenster — als Randnotiz für ein etwaiges v2 dokumentiert (EV-138).

## 20. Landingpage-Abnahmeplan

| Prüfung | Kriterium |
|---|---|
| Hauptaussage | Ein Besucher kann nach dem Hero in einem Satz sagen, was das Produkt tut (Kurztest mit 2-3 Personen, informell) |
| Claims | Jede Aussage hat Produktbeleg (Checkliste aus EV-121/122); D1–D3 behoben |
| Kernnutzen sichtbar | eigene Analyselogik, Vergleich, Kurse+Historie jeweils in Value/Panel adressiert |
| CTAs | primär Registrierung (Hero+Final), Login sekundär, Routing korrekt; mobil sichtbar ohne Scroll-Falle |
| Responsive | Mobile/Tablet/Desktop ohne Überläufe; svh-Hero intakt |
| Eingeloggt/ausgeloggt | `/` leitet eingeloggt um; Header zustandsabhängig korrekt |
| SEO | `lang="de"`, ein h1, Title+Description+OG, Lighthouse-SEO ≥ 90 |
| Analytics/Consent | Beacon weiterhin consent-gated; `?src=`-Attribution funktioniert; keine neuen Cookies |
| Conversion-Messung | `product_events` zeigt `user_registered` mit `src`; Vergleichszeitraum 4 Wochen vor/nach Umstellung im Admin-Dashboard auswertbar (Scrolltiefe/Abbruchpunkte sind ohne Zusatz-Tracking nicht messbar — bewusster Verzicht, E8) |

## 21. Risiken und Schutzmaßnahmen

| Risiko | Wahrscheinlichkeit | Wirkung | Schutz |
|---|---|---|---|
| Code-Splitting bricht eine Route/einen Deep-Link | mittel | hoch | vollständiger Routen-Smoke-Test (EV-114), ein revertierbarer Commit, `lazyWithRetry` gegen Stale-Chunks |
| priceStore-Ref-Counting-Bug (Poll läuft weiter/stoppt zu früh) | mittel | mittel | StrictMode-Doppelmount-Test, Unmount-Test, Signatur unverändert → kleiner Blast-Radius |
| currentUserCache liefert veralteten User (Rolle/Plan) | niedrig | hoch (AdminRoute!) | kurze TTL, explizite Invalidierung, AdminRoute-Sonderbehandlung (EV-112) |
| Guard-Zentralisierung ändert Fehlertexte | niedrig | mittel | Fehlertext-Snapshots in EV-101, textidentische Rückgaben (EV-130) |
| ttm-Delegation berührt versehentlich Annual-Branch | niedrig | kritisch | Golden Master vor/nach jedem Schritt; Delegation als zusätzlicher Zweig, nie Umbau des bestehenden |
| Alte Clients/Bookmarks senden quarterly, neue senden ttm → Verwirrung in History | niedrig | niedrig | frequency wird pro Eintrag gespeichert und angezeigt; keine Umbenennung |
| Rollback nach ttm-Release: localStorage enthält „ttm" | mittel | niedrig | Whitelist-Fallback beim Load (EV-134) |
| Landingpage-Umformulierung schwächt SEO-Rankings | niedrig | niedrig | h1-Struktur bleibt, Title/Description kommen hinzu (netto positiv) |
| Partikel-Ersatz wirkt auf Mobile „billig" | niedrig | niedrig | Gradient sorgfältig annähern, Betreiber-Review vor Merge (EV-110) |
| Cloudflare/Consent-Verhalten durch `?src=` berührt | — | — | kein Client-Tracking, nur Query-Param im eigenen Funnel — kein Consent-Bezug |

## 22. Migrations- und Rollback-Strategie

- **Keine Datenmigration nötig:** DB-Spalte `analysis_history.frequency` String(20) nullable; Custom-Definitionen-JSON bleibt; Compare-localStorage wird defensiv gelesen. Es wird bewusst **keine** Alembic-Migration erstellt.
- **Kein Cache-Umbau:** ttm berechnet nichts Neues → bestehende `*_quarterly`- und `*_ttm_*`-Artefakte werden wiederverwendet; keine Invalidierung, keine Kollisionen (Key-Räume unverändert).
- **Versionierung:** nicht erforderlich in v1 — `frequency` im History-Eintrag genügt zur Reproduzierbarkeit, da ttm keine neue Numerik einführt. Falls das Follow-up „annual → echte Jahresdaten" (EV-138) später kommt, wird dort eine `period_calculation_version` erwogen (dokumentierte Vorentscheidung, nicht jetzt bauen).
- **Rollback-Pfade:** TTM: `resolve_ttm_alias()` in `agent/frequency.py` so ändern, dass sie "ttm" NICHT mehr übersetzt (ein Zweizeiler), plus "ttm" aus `TTM_CAPABLE_METRICS` und der Katalog-`enum_values` entfernen (Backend lehnt "ttm" wieder ab); Frontend-Toggle-Option/Default zurücknehmen. Gespeicherte ttm-History-Einträge bleiben lesbar (Anzeige „TTM"), ein Rerun würde nach dem Rollback den Ablehnfehler zeigen — akzeptiert, kein Datenverlust. Compare-localStorage-Whitelist (`useCompare.tsx`) fällt bei unbekanntem Wert ohnehin defensiv auf „annual" zurück. Performance: jede Maßnahme ist ein eigener revertierbarer Commit. Landingpage: String-Reverts.
- **Deploy-Reihenfolge TTM:** Backend zuerst (akzeptiert ttm, Frontend sendet es noch nicht), dann Frontend — vermeidet 4xx-Fenster.
- **Tatsächliche Umsetzung (2026-07-16):** Statt eines einzigen erweiterten `ALLOWED_FREQUENCIES`-Tupels wurde "ttm" als Übersetzung `resolve_ttm_alias()` direkt in den 15 TTM_CAPABLE_METRICS-Methoden (Model.py) sowie in `analyze_wachstumswerte` (AgentAction.py) verankert — jede Methode übersetzt "ttm"→"annual" VOR ihrem eigenen Guard/Cache-Key, sodass `ALLOWED_FREQUENCIES`/`MODE_ALLOWED_FREQUENCIES` selbst zweiwertig bleiben konnten (siehe `agent/frequency.py`). Das vereinfacht den Rollback zusätzlich: ein Revert der `resolve_ttm_alias`-Aufrufe an diesen ~16 Stellen genügt, ohne dass irgendein Guard in DataLoader.py/sec_source.py angefasst werden musste.

## 23. Definition of Done

**Performance:** Baseline erhoben und dokumentiert (§7 gefüllt); P1–P6 adressiert; Entry-Bundle-Ziel erreicht; `/me`=1 und `/current-price`=eindeutige Symbole nachgewiesen; kein Partikel-rAF auf Mobile; CLS < 0,1 Dashboard; Desktop nicht verschlechtert; jede Maßnahme mit Vorher/Nachher-Werten protokolliert.

**Landingpage:** D1–D3 behoben; Copy-Deck freigegeben und 1:1 umgesetzt; SEO-Fixes live (Lighthouse-SEO ≥ 90); `?src=`-Attribution in `product_events` sichtbar; §20-Abnahmeplan vollständig abgehakt; keine unbelegbaren Aussagen.

**TTM:** Golden Master annual+quarterly diff-frei über alle Phasen; ttm für alle ✓-Metriken verfügbar und == annual belegt; Capability-Ausschlüsse wirksam (Backend-Fehler + UI-Ausblendung); Defaults umgestellt (frisch=ttm, Bestand unangetastet); Edge-Case-Suite grün; Plausibilitätsrechnung dokumentiert; Rollback per Kill-Switch verifiziert (einmalig lokal getestet: „ttm" aus ALLOWED → sauberer Fehler).

**Gesamt:** alle Aufgaben EV-100–EV-139 abgeschlossen oder begründet zurückgestellt; `npm run build` + pytest + vitest grün; EV-139-Protokoll in diesem Dokument.

## 24. Abschließende Verifikations-Checkliste

Stand 2026-07-16 (Sonnet 5, nach Umsetzung EV-101/110-114/120-124/130-138):

- [x] Golden Master (annual+quarterly) grün, 3 unabhängige Läufe diff-frei protokolliert (§ Umsetzung EV-101)
- [x] `pytest agent/tests api/tests` vollständig grün — 328 Tests (297 Bestand + 31 neu aus diesem Vorhaben)
- [x] `npm run build` grün (echter Build, nicht nur `tsc --noEmit`) — Entry-Chunk 349 KB → **138 KB gzip** (EV-114-Ziel ~130 KB nahezu erreicht)
- [ ] Alle 22 Routen manuell im Browser durchgeklickt — **nur Dev-Server-Stichprobe** (Landing, Login mobil+desktop) verifiziert, kein authentifizierter Durchlauf aller `/app/*`-Routen (kein Test-Login/DB in dieser Session verfügbar)
- [ ] Analyze/Custom/Compare je mit annual, quarterly, ttm im echten Browser durchgespielt — **stattdessen** durch Backend-Delegationstests (`test_ttm_delegation_ev132.py`, `test_ttm_mode_methods_ev135.py`) und Frontend-Typecheck/Build abgedeckt, keine authentifizierte UI-Session
- [x] Alte Custom-Definition (quarterly) lädt und rechnet unverändert — Golden Master + `test_quarterly_is_unaffected_by_ttm_delegation` beweisen dies auf Berechnungsebene
- [ ] History-Rerun für alte annual/quarterly-Einträge und neuen ttm-Eintrag im echten Browser — nicht live getestet (kein Test-Account)
- [x] Network-Nachweis: 1× `/me`, N× `/current-price` (N = eindeutige Symbole) — durch Code-Review von `currentUserCache.ts`/`priceStore.ts` sichergestellt; **nicht** per Live-Network-Tab bei eingeloggtem Nutzer verifiziert
- [x] Mobile-Emulation: kein Partikel-Canvas (Browser-Screenshot mobil bestätigt statischen Gradient), keine framer-Infinite-Loops in Badges (Code-Review: CSS-Keyframe ersetzt `motion.span`)
- [ ] Lighthouse Mobile + Desktop Abschlusswerte in §7 eingetragen — **EV-100/EV-117 (Baseline-Messung) wurden nicht durchgeführt**, siehe offener Punkt unten
- [ ] Landingpage-Abnahmeplan §20 vollständig abgehakt — Texte/SEO/CTA-Attribution umgesetzt und getestet, aber keine informelle Nutzerverständlichkeits-Stichprobe durchgeführt
- [x] CTA-Attribution: `test_register_cta_attribution_ev124.py` beweist `src`-Metadata am `user_registered`-Event auf DB-Ebene (In-Memory-SQLite, kein Live-Postgres)
- [x] Kill-Switch-Pfad dokumentiert und architektonisch verifiziert (§22): `resolve_ttm_alias`-Revert + `TTM_CAPABLE_METRICS`-Leerung; nicht als separater manueller Durchlauf ausgeführt, da äquivalent zum bereits bewiesenen Golden-Master-Diff-Verhalten

**Offen/nicht in dieser Session erledigt:** EV-100 (Performance-Baseline mit echtem Lighthouse/realen Geräten), EV-115/116/117 (Skeletons, gezielte Memoisierung, Nachmessung) — diese benötigen eine laufende, authentifizierte Umgebung mit Testnutzer bzw. echte Lighthouse-Durchläufe, die in dieser Sitzung nicht verfügbar waren. EV-125 (optionaler Track-Endpoint) bewusst zurückgestellt (E8).
- [ ] Keine Änderungen an package.json/Lockfiles, keine Alembic-Migration, kein Cache-Umbau (git diff-Nachweis)

---
---

# Internationalisierung – Deutsch & Englisch

Stand: 2026-07-18 (Fable 5, Planungsphase). Dieses Kapitel dokumentiert den vollständigen, freigegebenen Plan zur Internationalisierung der App (DE + EN). Die Umsetzung erfolgt phasenweise durch Sonnet 5 nach dem Sicherheitsprinzip in § 22.

## 1. Ziel und Hintergrund

Die gesamte App (öffentliche Seiten + eingeloggte App) soll zusätzlich vollständig auf Englisch verfügbar sein. Die deutsche Version bleibt funktional, strukturell und visuell 1:1 erhalten. Auf der AccountPage kommt eine Spracheinstellung hinzu. Architektur von Anfang an so, dass weitere Sprachen (fr, es, …) ohne Umbau ergänzbar sind.

**Bestätigte Produktentscheidungen (Betreiber, 2026-07-18):**
- Default-Sprache für neue/unbekannte Besucher: **Deutsch** (Browser `en-*` → automatisch Englisch)
- Englische Locale: **en-US** (SEC-Daten, NYSE/NASDAQ, US-Terminologie; Datum MM/DD/YYYY)
- **URLs bleiben unverändert** — keine `/de`/`/en`-Prefixe (EN = UI-Feature ohne eigenes SEO)
- Scope v1: **App + öffentliche Seiten.** Admin bleibt deutsch, System-E-Mails eigene spätere Phase, Legal-Seiten bleiben deutsch bis zur fachlich/rechtlich geprüften Übersetzung

## 2. Nicht verhandelbare Anforderungen

1. Deutsch funktioniert exakt wie vorher (Strings werden **verbatim** verschoben, nie umformuliert; DE-Formatter-Ausgaben byte-identisch).
2. Englisch besitzt exakt dieselben Funktionen wie Deutsch.
3. Sprachwechsel verändert keine Daten, Analyseergebnisse, Custom Analyses, Subscriptions oder technischen IDs.
4. Sprache und Währung sind vollständig getrennt (Währung = ISO-Code vom Backend, Locale nur Darstellung).
5. Bestehende Nutzer verlieren keine Einstellungen (`users.locale` nullable, NULL = heutiges Verhalten).
6. Weitere Sprachen später ohne Architekturumbau.
7. Keine Big-Bang-Migration; jede Phase unabhängig shipbar und rücknehmbar.

## 3. Bestehende Architektur

- **Frontend:** React 19.2 + TS 5.9 + Vite 7.2, react-router-dom 7, recharts, framer-motion, Sentry, vitest (nur 2 Testdateien). SPA ohne SSR; Seiten lazy via `lazyWithRetry`.
- **State:** kein Redux/React Query; Context-Provider-Stack in `App.tsx` (Compare, AnalysisJobs, AnalyzeWorkspace, Favorites, ThemeMode), `ToastProvider` in `main.tsx`. **Kein Auth-Context** — User via Modul-Cache `getCurrentUser()` in `frontend/src/api/auth.ts` (30 s TTL, invalidiert per Window-Events `app:login`/`app:logout`).
- **Backend:** FastAPI + SQLAlchemy 2 + PostgreSQL + Alembic (21 Migrationen). `api/models/user.py` ohne locale/preferences-Feld. `PATCH /auth/profile` existiert.
- **SEO:** statisches deutsches `index.html` + vorgerendertes Glossar (`frontend/scripts/generate-glossary.ts` → `dist/glossar/`, liest aus `metricsConfig.ts`).
- **Persistenz-Muster vorhanden:** `theme-mode` in localStorage.

## 4. Bestehende Internationalisierungsmechanismen

**Keine.** Kein i18n-Paket, kein Language-Context, keine Übersetzungsdateien. `<html lang="de">` statisch. Nur hartkodierte `toLocaleString("de-DE")`-Aufrufe (verstreut) und zwei **inkonsistente** `formatCompactNumber`-Implementierungen: `chartUtils.ts` (Tsd./Mio./Mrd., de-DE) vs. `metricFormatting.tsx` (K/M/B). Wiederverwendbar: theme-mode-localStorage-Muster, `app:login`/`app:logout`-Events, `invalidateCurrentUserCache()`.

## 5. Vollständige String-Inventur

Gesamtschätzung: **~900–1.100 Frontend-Strings** + **~100–200 nutzersichtbare Backend-Texte**. Kategorien: UI, Navigation, Form, Validation, Error, Success, Analysis, Financial Metric, Chart, Tooltip, Account, Billing, Authentication, Marketing, SEO, Legal, Accessibility, Backend, Admin.

### 5.1 Common UI
Toast-System (`components/ui/Toast.tsx`, ~53 inline `showToast()`-Aufrufe verstreut), Modal (`components/ui/Modal.tsx`, aria „Schließen"), ErrorBoundary (~2), Cookie-Banner (`components/consent/CookieConsentBanner.tsx`, ~13), Suspense-Fallbacks. Keine 404-Seite (Fallback-Redirect auf `/`).

### 5.2 LandingPage
`pages/public/LandingPage.tsx` ~60–80 Strings (Marketing-Copy in Daten-Arrays + JSX-Props); `components/landing/AudienceTabs.tsx` ~7; `components/intro/IntroSlogan.tsx` ~1–3; `pages/public/PricingPage.tsx` ~12 + `config/pricingPlans.ts` (~14, inkl. „Spare 16,7 %", „2 Monate gratis").

### 5.3 Authentication
Login (~9), Register (~10), ForgotPassword (~5), ResetPassword (~4–8), VerifyEmail (~8) + `EmailVerificationBanner` (~3). Validierungstexte inline in den Seiten.

### 5.4 Dashboard
`DashBoardPage.tsx` ~14 + `DashboardFavoritesSection.tsx`; Datumsformate `toLocaleDateString("de-DE")`.

### 5.5 AnalysisPage
`AnalyzePage.tsx` ~18 + `components/analysis/*` (FrequencyToggle mit EN-Labels „Annual/TTM/Quarterly", AnalysisFrequencyPanel „X/Y erfüllt"/„Erfüllt/Kritisch/Neutral", DossierDetailPanel/DossierSummaryCard rendern Backend-Texte, QuotaExceededModal, SymbolCommandField).

### 5.6 CustomAnalysisPage
Kein eigener Route-Screen (in Analyze eingebettet): `components/customAnalysis/*` (6 Dateien, ~25 Strings) + `hooks/useCustomAnalysisDefinitions.ts` (Toasts „Analyse gespeichert." etc.).

### 5.7 CompareAnalysisPage
`ComparePage.tsx` ~5 + `components/compare/*` + `hooks/useCompare.tsx` (~6, inkl. Template-Strings).

### 5.8 AccountPage
`pages/app/AccountPage.tsx` (~1707 Zeilen, ~44 Strings): Hero, Kontoinformationen, Mitgliedschaft, Profil, Kündigen-Modal, Passwort, Zahlungen, Danger Zone. Label-Maps `getPlanLabel`/`getBillingStatusLabel`/`getBillingIntervalLabel` (Z. ~1151–1196).

### 5.9 BillingPage
`BillingPage.tsx` ~12 + `pages/billing/SuccessPage.tsx`/`CancelPage.tsx` (~7).

### 5.10 Navigation
Zentral: `components/layout/AppSidebar.tsx` (Nav-Array Z. 35–46: Dashboard/Analyse/Vergleich/Konto/Support/Abrechnung/Admin, inkl. Mobile-Nav + Account-Menü + Toasts), `Header.tsx` (~5), `Footer.tsx` (~3).

### 5.11 Admin
`AdminDashboardPage.tsx` (~8) + `components/admin/*` (~27). **Produktentscheidung: bleibt deutsch** (Betreiber-intern), kein i18n in v1.

### 5.12 Errors & Validation
`utils/jobErrors.ts` (JOB_LOST_MESSAGE), `api/client.ts` (Fallback „Anfrage fehlgeschlagen…", reicht Backend-`detail` 1:1 durch), `api/auth.ts` („Nicht eingeloggt."), Formular-Validierung inline pro Seite. **Backend:** `HTTPException.detail` gemischt DE/EN (auth.py „Benutzername bereits vergeben", billing.py, metric_routes.py, symbol_validation.py ~17 deutsche, custom_analysis.py, favorites.py, dependencies.py).

### 5.13 SEO & Metadata
Statisches `index.html` (deutscher Title/Description/OG). Keine Laufzeit-`document.title`-Setzung. Glossar-Seiten deutsch (bleiben deutsch).

### 5.14 Accessibility
~19 `aria-label`, ~32 `placeholder`, ~39 `title=` verstreut; keine `alt=` (SVG-Icons). `<html lang>` statisch.

## 6. Technische IDs vs. Anzeigenamen

Sauber getrennt, mit **einer Ausnahme**:
- Frequency `"annual"|"quarterly"|"ttm"` (`types/frequency.ts`, `agent/frequency.py`) — stabil, Anzeige separat.
- Metric-Keys: snake_case-IDs als Objekt-Keys/API-Parameter/persistierte Werte; `normalizeMetricKey` lowercased fürs Config-Lookup, der **rohe Katalog-Key** (teils gemischtes Casing wie `calculate_KGV`) bleibt die übertragene ID.
- Operatoren: Symbol-IDs `">" | "<" | ">=" | "<="`.
- Plan/Billing-Status: `free/friends/pro/admin`, `active/canceling/past_due/canceled/payment_failed_canceled`, `month/year` — Anzeige über Label-Maps.
- Analytics: backend-seitige `product_event`s mit englischen snake_case-IDs + ID-Metadata. Kein Handlungsbedarf; optional später `locale`-Property.
- ⚠️ **GESCHÜTZTER LEGACY-KEY:** `api/routes/analyze.py:270` baut Result-Dict-Keys `f"{DISPLAY_NAME[mode]}|{frequency}"` (z. B. `"Wachstumswerte|annual"`); Frontend parst in `types/analysis.ts:39–51`. **Wird eingefroren, niemals lokalisiert** — Kommentar-Marker an beiden Stellen, Anzeige über inverse Map `DISPLAY_NAME-Teil → mode-ID → t("analysis.modes.<id>")`.

## 7. Finanzterminologie und Glossar

Zentrale Terminologie-Quelle = `metrics`-Namespace (ein Eintrag pro Metric-Key mit `label`/`description`/`formula`). Eine Kennzahl hat app-weit genau eine Übersetzung (kein „Umsatz/Erlös/Einnahmen"-Drift möglich, weil alle Stellen über `getMetricLabel(key)` gehen). Heute schon englische DE-Labels („ROE", „Profit Growth") bleiben im DE-Dictionary wörtlich erhalten (Parität). EN-Terminologie: US-Standard (Revenue, Net Income, TTM = „Trailing Twelve Months"; DE-Anzeige „TTM/Letzte 12 Monate" wie bisher im UI). Perioden: `annual` → DE „Annual" (heutiges Label bleibt!) / EN „Annual"; die heutigen englischsprachigen Frequenz-Labels im DE-UI bleiben unverändert (Parität schlägt Eindeutschung).

## 8. Dynamische Texte

- **Backend-generiert** (`agent/AgentAction.py`): `overall_assessment` (endliche Wertemenge, quasi-Codes) + `message` (f-String-interpoliert, gemischt DE/EN). Strategie: **DE = Passthrough (byte-identisch), EN = Frontend-Rekonstruktion** aus Strukturdaten (`value`, `meets_criterion`, Schwellen, Metric-Labels) via `renderCriterionMessage()`; unbekannte Werte → Passthrough. Langfrist-Option (nicht jetzt): additives `message_code` + `message_params`.
- **Frontend-generiert:** Score-Texte („{n}/{total} erfüllt", AnalysisFrequencyPanel), Template-Strings in AccountPage, useAnalysisJobs, useCompare, DashBoardPage, DossierDetailPanel → strukturierte Templates mit `{var}`-Interpolation, keine String-Konkatenation. Pluralisierung über natives `Intl.PluralRules`.

## 9. Locale Formatting

Neues Modul `frontend/src/i18n/format.ts`; Mapping `de → "de-DE"`, `en → "en-US"`. **Regel: DE-Pfade sind wörtliche Kopien der heutigen Implementierungen**, vor Umbau per Characterization-Tests eingefroren.

### 9.1 Numbers
`formatNumber(value, locale, opts)` via `Intl.NumberFormat`. Kompakt: `formatCompactNumberChart` (DE-Zweig = heutige chartUtils-Logik Tsd./Mio./Mrd. mit identischen Schwellen; EN-Zweig K/M/B, gleiche Schwellen) und `formatCompactNumberMetric` (heutige K/M/B-Logik, bleibt für DE **unverändert K/M/B** — die bestehende Inkonsistenz wird bewusst NICHT „mitrepariert").
### 9.2 Percentages
`formatPercent(value, locale, decimals)` — DE `12,4 %` (heutiges Format inkl. Leerzeichen), EN `12.4%`.
### 9.3 Dates
`formatDate(iso, locale, style)` — DE-Pfad exakt heutige `toLocaleDateString("de-DE", …)`-Optionen; EN en-US.
### 9.4 Currencies
`formatCurrency(value, currencyISO, locale)` — Währung kommt ausschließlich als ISO-Code vom Backend (`reporting_currency`/`currency`-Felder); Locale ändert nur Darstellung, **niemals** die Währung. Abo-Preise bleiben fix `€`.

## 10. Spracheinstellung und Persistenz

- **DB:** Alembic-Migration `users.locale VARCHAR(10) NULL`, kein Default, kein Backfill. NULL = keine Präferenz → Bestandsnutzer exakt wie heute. Rollback = Spalte droppen, verlustfrei.
- **API:** bestehendes `PATCH /auth/profile` + `UserResponse` additiv um `locale` erweitert; Server validiert ∈ {"de","en"} sonst 422. Kein neuer Endpunkt.
- **Prioritätslogik:** `user.locale` (eingeloggt, gültig) → `localStorage["app-locale"]` → `navigator.language(s)` (Prefix-Match, „en-GB" → „en") → `"de"`. Jeder Schritt validiert gegen `SUPPORTED_LOCALES`; ungültige Werte fallen lautlos durch.
- **Login** (`app:login`-Event): gesetzte `user.locale` gewinnt über Gerätewahl und wird in localStorage gespiegelt; NULL → Gerätewahl bleibt, **kein automatischer Profil-Write** (persistiert wird nur bei expliziter Wahl).
- **Logout:** localStorage + aktive Sprache bleiben. **Neues Gerät:** navigator → nach Login User-Pref.
- **AccountPage-UI:** eigene kleine Karte „Sprache / Language", Segmented Control „Deutsch | English" (Optik analog FrequencyToggle, inline CSSProperties). Optimistic: Klick → sofort `setLocale` (Context + localStorage) → parallel `PATCH /auth/profile {locale}` → Erfolg: `invalidateCurrentUserCache()`; Fehler: Toast-Warnung, UI bleibt umgeschaltet. Ab ≥4 Sprachen → Dropdown.

## 11. Verhalten für ausgeloggte Nutzer

localStorage → navigator.language → Deutsch. Kompakter DE/EN-Switcher im Header von PublicLayout **und** AuthLayout (schreibt nur localStorage + Context, kein API-Call). Landing/Pricing/Auth/Legal damit umschaltbar (Legal-Inhalte selbst bleiben v1 deutsch, nur UI-Chrome wechselt).

## 12. Routing- und SEO-Strategie

URLs bleiben unverändert (bestätigt) — kein Migrationsrisiko für Links, Bookmarks, Login-/OAuth-/Stripe-Redirects, Analytics. Statisches `index.html` bleibt deutsch → deutsche SEO exakt erhalten. Laufzeit: Hook `usePageMeta(titleKey, descKey)` setzt `document.title` + Meta-Description pro Route+Locale; `<html lang>` dynamisch. **Dokumentierter Trade-off:** ohne getrennte URLs kein `hreflang`, EN-Inhalte nicht separat indexierbar — EN ist UI-Feature, kein SEO-Kanal. Falls EN-SEO später Ziel wird: `/en/*`-Prefix als eigenes Projekt (Routing, Canonicals, Prerender). Glossar bleibt deutsch.

## 13. Zielarchitektur i18n

**Leichtgewichtiges typisiertes Eigenbau-Dictionary** (keine Library). Begründung: (1) Paritätsanforderung dominiert — DE-Dictionary ist wörtlicher Copy-Paste des heutigen Codes, keine Library-Fallback-Magie kann DE-Ausgaben verändern; (2) stärkste Typsicherheit: DE als `as const` → Schema, EN typisiert dagegen → fehlender Key = **Compile-Fehler**; (3) ICU-Bedarf gering (`{var}`-Interpolation ~20 Zeilen, Plural via `Intl.PluralRules`, 0 KB); (4) ~0,5 KB statt 12–17 KB gzip; (5) Hook-API i18next-kompatibel (`useTranslation(ns)` → `{t}`) als Exit-Strategie. Kipppunkte für Library-Wechsel: >4 Sprachen, externes TMS, komplexe ICU-Selects.

Neue Module unter `frontend/src/i18n/`:
```
config.ts             SUPPORTED_LOCALES=["de","en"], Locale-Typ, DEFAULT_LOCALE="de",
                      LOCALE_STORAGE_KEY="app-locale", INTL_LOCALES {de:"de-DE", en:"en-US"}
localeContextValue.ts Context + Typ (Repo-Muster analog Toast)
LocaleProvider.tsx    State, <html lang>-Sync, DE eager / EN lazy via import()
useTranslation.ts     useLocale(), useTranslation(ns) → { t, locale }
t.ts                  interpolate(template, params), plural(locale, count, forms)
detect.ts             Prioritätslogik
format.ts             locale-aware Formatter (§ 9)
apiErrorMessages.ts   detail-String/code → Translation-Key (Phase 11)
locales/de/*.ts       Namespaces; locales/en/*.ts typisiert gegen DE-Schema
```
`LocaleProvider` in `main.tsx` außerhalb `ToastProvider`. Sprachwechsel = reiner React-State → Re-Render ohne Reload; App-State (Compare-Auswahl, laufende Jobs, Workspace, Favoriten, Router, Formulare) bleibt per Konstruktion erhalten, weil Locale nirgends in Request-Bau/State-Keys einfließt. Bis EN-Bundle geladen ist bleibt DE sichtbar (nie rohe Keys).

## 14. Translation-Key-Architektur

Namespaces: `common, nav, auth, landing, pricing, dashboard, analysis, customAnalysis, compare, account, billing, support, errors, validation, metrics, charts, tour, agent` (später: `legal`, `admin`). Keys semantisch + camelCase (`account.profile.editButton`, nicht `text1`/`account.bearbeiten`), verschachtelte Objekte statt Dot-Strings (TS-Inferenz), Interpolation `{name}`, kein HTML in Strings (Satz-Segmente `before/linkLabel/after`). DE eager, **EN als ein Lazy-Bundle** (bei 2 Sprachen ausreichend; Struktur bleibt splitting-fähig). `metrics`-Sonderfall: `METRICS_CONFIG` behält Technik (unit/decimals/scale), `label/description/formula` wandern nach `locales/de/metrics.ts` (per Extraktionsskript 1:1 generiert); `generate-glossary.ts` liest künftig daraus, Output muss **byte-identisch** bleiben (Snapshot).

## 15. Missing Translation & Fallback Strategy

Fallback-Kette: `en → de → Key-Rohwert`. Compile-Zeit: EN-Module sind gegen das DE-Schema typisiert (`TranslationShape<typeof de>`) → fehlender/überzähliger Key = TS-Fehler. CI: vitest `completeness.test.ts` vergleicht rekursiv die Key-Mengen (fängt leere Strings). Laufzeit: Dev-`console.warn` bei Missing Key + DE-Fallback — Produktion zeigt **nie** rohe Keys wie `analysis.chart.revenue.title`.

## 16. Schutz bestehender Business Logic

Sprachwechsel darf nicht ändern: Company Symbol, Analysis Mode ID, Metric ID, Period ID, Subscription ID, User ID, Saved Analysis ID, Custom-Definition, API-Parameter, Cache-Identity. Nachweis: Paritätstests (§ 28) rendern Kernflows unter beiden Locales mit gemocktem fetch → Request-URLs/Bodies deep-equal; Result-Key-Parser liefert identische Mode-IDs; Datenwerte in Charts identisch (nur Labels/Formatierung unterscheiden sich).

## 17. Custom Analysis Compatibility

Persistiert werden nur technische Keys, Operator-Symbole, numerische Thresholds und der **nutzergewählte freie `name`**. User-generated Content („Meine Value Strategie") wird **niemals** übersetzt. Test: persistierte Definition vor/nach Sprachwechsel byte-identisch; Speichern unter EN erzeugt identische DB-Rows wie unter DE.

## 18. API- und Backend-Kompatibilität

- **HTTPException-Texte:** rein frontendseitige Mapping-Schicht `apiErrorMessages.ts` — `ApiError.code` vorhanden (Mechanismus existiert: `QUOTA_EXCEEDED`, `EMAIL_NOT_VERIFIED`) → Key; sonst exakter String-Match auf Katalog bekannter deutscher detail-Literale; sonst Roh-detail (= Status quo, DE-Parität automatisch; EN-Nutzer sieht schlimmstenfalls deutschen Text, kein Bruch). Später additiv: Backend hebt Strings schrittweise auf `detail={"code":…, "message":"<identischer deutscher Text>"}`. **Serverseitiges `Accept-Language`: verworfen** (doppelter Katalog, Paritätsnachweis erschwert, Cache-Komplexität).
- **AgentAction-Sätze:** Backend unangetastet (§ 8).
- Kein API-Umbau; alle Erweiterungen additiv; kein bestehender Client bricht.
- E-Mails (`api/services/email_service.py`, inline f-Strings, gemischt EN/DE): eigene spätere Phase; `user.locale` steht danach als Template-Selektor bereit.

## 19. Performance-Auswirkungen

DE-Strings liegen heute schon im Bundle (inline) — Verschiebung in eager DE-Dictionaries ist größenneutral (± Overhead der Objektstruktur). EN als ein lazy Chunk (~ Größe der DE-Texte, geschätzt < 100 KB roh, deutlich kleiner gzip), wird nur bei EN-Aktivierung geladen. Kein zusätzlicher Provider-Rerender-Hotspot: Locale-Context ändert sich nur beim expliziten Wechsel. Keine zukünftigen Sprachdateien im Initial-Load. `Intl.*` ist nativ (0 KB). Entry-Chunk-Budget (138 KB gzip, § oben EV-114) wird nach Phase 11 (Landing) per `npm run build` nachgemessen.

## 20. AccountPage Profil Spacing Bug

Siehe eigener Abschnitt „AccountPage Profil Spacing Bug" am Dokumentende (UI-001).

## 21. Offene Produktentscheidungen

Alle vier Kernfragen sind entschieden (§ 1). Verbleibend offen (blockiert nichts):
1. **EN-Copy-Review:** Wer prüft die englischen Marketing-/Analyse-Texte fachlich? (Empfehlung: Betreiber-Review pro Phase; maschinelle Unterstützung erlaubt, ungeprüfte Übernahme nicht — besonders Finanzbegriffe, Billing, CTAs.)
2. **Legal-Übersetzung:** Zeitpunkt + juristische Prüfung (eigener Track, nicht v1).
3. **E-Mail-Zweisprachigkeit:** Zeitpunkt (eigener Track; danach auch Vereinheitlichung der heute gemischten EN/DE-Mails sinnvoll).
4. **Analytics-`locale`-Property** an product_events (optional, additiv).

## 22. Umsetzungsphasen

**Sicherheitsprinzip pro Phase (verbindlich für Sonnet 5):** 1. Ist-Verhalten dokumentieren/screenshotten → 2. Tests ausführen → 3. nur Text-/Localization-Schicht ändern (Strings verbatim verschieben) → 4. Tests erneut → 5. DE visuell prüfen → 6. EN visuell prüfen → 7. Funktionsparität prüfen (Network-Requests identisch). Vor „fertig": echtes `npm run build`. Erst dann nächste Phase.

### Phase 0 – Baseline und Regression absichern
Characterization-Tests der heutigen Formatter-Ausgaben (chartUtils.formatCompactNumber, formatPercentChange, metricFormatting.formatMetricValue/formatCompactNumber) über Wertekatalog als Literal-Snapshots festschreiben; Glossar-HTML-Snapshot; DE-Screenshots der Hauptseiten.
### Phase 1 – i18n Foundation
`i18n/`-Gerüst komplett (config, t, LocaleProvider, useTranslation, detect, format.ts, leere de/en-Namespaces, completeness-Test), Delegation chartUtils/metricFormatting → format.ts (DE-Pfad, Tests aus Phase 0 bleiben grün), Provider in main.tsx. **Noch kein String migriert — nichts rendert anders.**
### Phase 2 – Locale Preference und Account Settings
Alembic-Migration, Model/Schema/Route additiv, `auth.ts`-Typen, Sprach-Karte AccountPage, `app:login`-Sync, DE/EN-Switcher Public-/Auth-Header.
### Phase 3 – Common UI und Navigation
`common`, `nav`: AppSidebar (inkl. Mobile), Header, Footer, CookieConsentBanner, Toast-/Modal-Chrome, ErrorBoundary.
### Phase 4 – Authentication
5 Auth-Seiten inkl. Validierungen, Placeholder, EmailVerificationBanner.
### Phase 5 – Dashboard
DashBoardPage + Favoriten-Sektion, Datumsformate → formatDate.
### Phase 6 – AnalysisPage
**Höchstes Risiko.** metrics-Namespace + metricsConfig-Umbau + Glossar-Snapshot; Mode-Labels über inverse DISPLAY_NAME-Map (Legacy-Key einfrieren + Kommentar-Marker); Score-/Panel-Texte; Dossier-Renderer (DE-Passthrough / EN-Rekonstruktion).
### Phase 7 – CustomAnalysisPage
Builder-UI; user-Namen und persistierte Keys/Operatoren unangetastet.
### Phase 8 – CompareAnalysisPage
ComparePage, charts-Namespace, Empty-States, Tooltips.
### Phase 9 – Account und Billing
AccountPage-Resttexte, Label-Maps → t, BillingPage, Quota-Modals, Success/Cancel, Support.
### Phase 10 – LandingPage
Landing + Pricing (größte statische Textmenge), usePageMeta.
### Phase 11 – Errors, Validation und Dynamic Content
apiErrorMessages.ts, jobErrors, ~53 showToast-Stellen, Formular-Validierungen, Tour; Grep-Sweep auf verbleibende deutsche Literale.
### Phase 12 – SEO, Accessibility und Metadata
aria-labels, restliche placeholder/title, document.title/Meta aller Routen, `<html lang>`-Verifikation.
### Phase 13 – AccountPage Spacing Fix
**Vorgezogen umgesetzt am 2026-07-18** (unabhängig von i18n, eigener Commit) — siehe UI-001.
### Phase 14 – Full Regression & Verification
Matrizen §§ 25–28 abarbeiten, EN-Copy-Review, Doku-Abschluss.

Spätere eigene Tracks (nicht v1): Legal-Übersetzung, Admin-i18n, E-Mail-i18n.

## 23. Detaillierte Aufgaben

### [I18N-001] Formatter-Characterization-Tests einfrieren
**Status:** ✅ Umgesetzt (2026-07-18, Commit 2f10126) · **Phase:** 0 · **Priorität:** Kritisch · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** keine
**Ziel:** Heutige DE-Ausgaben aller Formatter als Literal-Snapshots festschreiben, bevor irgendetwas umgebaut wird.
**Aktueller Zustand:** `chartUtils.test.ts`/`metricFormatting.test.ts` decken Teile ab, nicht alle Grenzfälle.
**Fundstelle:** `frontend/src/components/charts/chartUtils.ts:114–127, 314–329`; `frontend/src/components/metrics/metricFormatting.tsx:25–39, 91–132`.
**Zu schützen:** exakte Strings inkl. Suffixe (Tsd./Mio./Mrd. bzw. K/M/B), Komma/Punkt, `±0,0 %`, `n. v.`, `Ja/Nein`, `—`.
**Änderung:** Neue Testdatei mit Wertekatalog (negativ, 0, Schwellen 10³/10⁶/10⁹ ±1, Dezimalgrenzen) und **erwarteten Literalwerten** (nicht berechnet).
**Schritte:** 1. Kataloge definieren 2. Ist-Ausgaben einmalig erzeugen und als Literale eintragen 3. Test grün committen.
**Tests:** vitest. **Akzeptanz:** [ ] Snapshots decken beide formatCompactNumber-Varianten, formatPercentChange, formatMetricValue ab. **Rollback:** Testdatei löschen (kein Produktivcode).

### [I18N-002] i18n-Kernmodul (config, t, Provider, detect)
**Status:** ✅ Umgesetzt (2026-07-18, Commit 2f10126) · **Phase:** 1 · **Priorität:** Kritisch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** —
**Ziel:** Tragfähiges, typisiertes i18n-Gerüst ohne sichtbare Verhaltensänderung.
**Änderung:** Dateien aus § 13; `LocaleProvider` um `<App/>` in `main.tsx`; `<html lang>`-Sync; EN-Lazy-Load mit `ready`-Flag.
**Schritte:** 1. config + t + Typen 2. detect (localStorage→navigator→"de") 3. Provider + Hook 4. leere Namespaces de/en + completeness-Test 5. Unit-Tests interpolate/plural/detect.
**Paritätstest:** App rendert unter DE pixelidentisch (kein String migriert). **Akzeptanz:** [ ] `npm run build` grün [ ] kein sichtbarer Unterschied. **Rollback:** Provider-Wrap entfernen, i18n/-Ordner löschen.

### [I18N-003] Formatting-Layer format.ts + Delegation
**Status:** ✅ Umgesetzt (2026-07-18, Commit 2f10126) · **Phase:** 1 · **Priorität:** Kritisch · **Aufwand:** M · **Risiko:** Mittel · **Abhängigkeiten:** I18N-001, I18N-002
**Ziel:** Eine zentrale locale-aware Formatter-Quelle; DE-Ausgaben byte-identisch.
**Zu schützen:** alle I18N-001-Snapshots; bestehende Tests bleiben inhaltlich unverändert grün.
**Änderung:** format.ts (§ 9); `chartUtils.formatCompactNumber`/`formatPercentChange` und `metricFormatting`-Helper delegieren mit fixem `"de"` (Aufrufstellen unverändert); `locale`-Parameter wird in späteren Phasen durchgereicht.
**Akzeptanz:** [ ] I18N-001-Snapshots grün [ ] chartUtils.test.ts/metricFormatting.test.ts grün. **Rollback:** Delegation revertieren (Originalcode bleibt bis Phase 14 als Referenz im Git-Verlauf).

### [I18N-004] users.locale: Migration + API
**Status:** ✅ Umgesetzt (2026-07-19) · **Phase:** 2 · **Priorität:** Hoch · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** —
**Änderung:** Alembic `users.locale VARCHAR(10) NULL`; `api/models/user.py`; `UserResponse` + Profile-Update-Schema additiv; Validierung ∈ {"de","en"} → 422; `frontend/src/api/auth.ts`-Typen.
**Zu schützen:** Bestehende User-Rows (NULL-safe), API-Verträge (nur additive Felder).
**Tests:** pytest PATCH mit gültig/ungültig/None; Migration up+down auf Kopie. **Akzeptanz:** [ ] Bestandsnutzer-Login unverändert [ ] Rollback per downgrade verlustfrei.

### [I18N-005] Spracheinstellung AccountPage + Login-Sync
**Status:** ✅ Umgesetzt (2026-07-19) · **Phase:** 2 · **Priorität:** Hoch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** I18N-002, I18N-004
**Änderung:** Sprach-Karte (Segmented Control) in AccountPage; Optimistic-Flow (§ 10); `app:login`-Listener im Provider.
**Manuelle Tests:** Wechsel DE↔EN eingeloggt, Reload, Logout/Login, zweites Gerät (Inkognito), Fehlerfall PATCH (Netz aus) → UI bleibt umgeschaltet + Toast.
**Akzeptanz:** [ ] Präferenz überlebt Reload + neues Gerät nach Login [ ] kein Zustandsverlust beim Wechsel.
**Verifiziert (2026-07-19):** Backend vollständig via pytest (`test_locale_profile_i18n_004.py`, 5 Tests + volle Suite 333 grün); Frontend `tsc -b`/Build/vitest grün, Header-Switcher live im Browser geprüft (Context/localStorage/`<html lang>` ohne Reload). **Nicht verifiziert:** der authentifizierte PATCH-Roundtrip der AccountPage-Sprachkarte im echten Browser — kein Backend-Server + kein verifizierter Testnutzer in dieser Session verfügbar (gleiche Einschränkung wie bei früheren Sessions, siehe § 24 oben). Nachholen, sobald ein Test-Login verfügbar ist.

### [I18N-006] DE/EN-Switcher Public-/Auth-Header
**Status:** ✅ Umgesetzt (2026-07-19) · **Phase:** 2 · **Priorität:** Hoch · **Aufwand:** S · **Risiko:** Niedrig · **Abhängigkeiten:** I18N-002
**Änderung:** kompakter Toggle in PublicLayout- und AuthLayout-Header; schreibt localStorage + Context.
**Responsive:** mobil sichtbar, Touch-Target ≥ 40 px. **Akzeptanz:** [ ] Landing/Login umschaltbar ohne Reload.

### [I18N-007] Common UI + Navigation migrieren
**Status:** ✅ Umgesetzt (2026-07-19) · **Phase:** 3 · **Priorität:** Hoch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** I18N-002
**Betroffene Dateien:** AppSidebar.tsx (Nav-Array + Mobile + Account-Menü), Header.tsx, Footer.tsx, CookieConsentBanner.tsx, Toast.tsx/Modal.tsx (aria), ErrorBoundary.tsx (ausgelagert nach ErrorFallback.tsx wegen Hook-in-Klassenkomponente), AppLayout.tsx (Mobile-Hamburger-aria-labels).
**Schritte:** 1. Strings verbatim nach `de/nav.ts`/`de/common.ts` 2. `t()`-Aufrufe 3. EN-Übersetzung 4. DE-Screenshot-Diff.
**Responsive:** Sidebar + Mobile-Nav mit längeren EN-Texten geprüft (Mobile-375px, Cookie-Banner nutzt eigene kürzere Mobile-Textvariante). **Akzeptanz:** [x] DE pixelidentisch (Header/Footer/Cookie-Banner Screenshot-Vergleich) [x] EN ohne Overflow (Desktop+Mobile geprüft).
**Verifiziert (2026-07-19):** `tsc -b`/ESLint/vitest (148 Tests)/`npm run build` grün. Browser: Landing-Header DE→EN→DE Wechsel ohne Reload, Footer-Links, Cookie-Banner (Desktop+Mobile-Variante) beide Sprachen, kein Konsolenfehler durch die i18n-Änderungen (ein vorbestehender, unabhängiger framer-motion-Style-Warning in `AuthLayout.tsx` — nicht in dieser Phase berührt). AppSidebar/AppLayout-Hamburger nur per Code-Review + Typecheck verifiziert (kein authentifizierter Testlauf verfügbar, gleiche Einschränkung wie I18N-005).

### [I18N-008] Auth-Seiten
**Status:** Offen · **Phase:** 4 · **Priorität:** Hoch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** I18N-007
Login/Register/Forgot/Reset/VerifyEmail + Banner inkl. Validierungstexte + Placeholder → `auth`/`validation`. Paritätstest: Login-Request-Body identisch DE/EN.

### [I18N-009] Dashboard
**Status:** Offen · **Phase:** 5 · **Priorität:** Mittel · **Aufwand:** M · **Risiko:** Niedrig-Mittel · **Abhängigkeiten:** I18N-003, I18N-007
DashBoardPage + Favoriten; `toLocaleDateString("de-DE")` → `formatDate`. Paritätstest: identische API-Calls; Kursdaten-Werte identisch.

### [I18N-010] Analysis: metrics-Namespace + metricsConfig-Umbau
**Status:** Offen · **Phase:** 6 · **Priorität:** Kritisch · **Aufwand:** L · **Risiko:** Hoch · **Abhängigkeiten:** I18N-003
**Ziel:** `label/description/formula` aus METRICS_CONFIG in `locales/de/metrics.ts` (per Extraktionsskript 1:1), Technik-Felder bleiben; `getMetricLabel(key)`-Helper; Glossar-Generator liest neu.
**Zu schützen:** Glossar-HTML **byte-identisch** (Snapshot-Vergleich); alle Metric-Key-Lookups unverändert (`normalizeMetricKey`).
**Akzeptanz:** [ ] Glossar-Diff leer [ ] alle Verbraucher (Analyze, Custom, Compare, Charts) zeigen DE unverändert.

### [I18N-011] Analysis: Legacy-Key-Schutz + Mode-Labels
**Status:** Offen · **Phase:** 6 · **Priorität:** Kritisch · **Aufwand:** S · **Risiko:** Hoch · **Abhängigkeiten:** I18N-010
**Änderung:** Kommentar-Marker „PROTECTED LEGACY KEY — niemals lokalisieren" an `api/routes/analyze.py:~270` und `types/analysis.ts:~39`; inverse Map DISPLAY_NAME→mode-ID; alle Anzeigen über `t("analysis.modes.<id>")`.
**Paritätstest:** Result-Fixture → Parser → identische Mode-IDs unter DE/EN. **Akzeptanz:** [ ] Wire-Format unverändert (Fixture-Bytes).

### [I18N-012] Analysis: Dossier DE-Passthrough / EN-Rekonstruktion
**Status:** Offen · **Phase:** 6 · **Priorität:** Kritisch · **Aufwand:** L · **Risiko:** Hoch · **Abhängigkeiten:** I18N-010, I18N-011
**Änderung:** `renderCriterionMessage()` + `agent`-Namespace + overall_assessment-Map (§ 8); DossierDetailPanel/SummaryCard; Score-Texte AnalysisFrequencyPanel → t.
**Zu schützen:** DE rendert **exakt** die Backend-Strings (Passthrough); unbekannte Werte → Passthrough auch unter EN.
**Paritätstest:** Fixture-Results je Analysemodus: DE-Rendering == heutiges Rendering (Snapshot); Datenwerte unter EN identisch. **Akzeptanz:** [ ] pro Analysemodus verifiziert; nicht abbildbare Sätze bleiben als Passthrough dokumentiert.

### [I18N-013] CustomAnalysis-UI
**Status:** Offen · **Phase:** 7 · **Priorität:** Hoch · **Aufwand:** M · **Risiko:** Mittel · **Abhängigkeiten:** I18N-010
6 customAnalysis-Komponenten + Definitions-Hook-Toasts. **Zu schützen:** persistierte Definition byte-identisch vor/nach Sprachwechsel; user-`name` nie durch t() geleitet. **Akzeptanz:** [ ] Save unter EN erzeugt identische DB-Row wie unter DE.

### [I18N-014] Compare + Charts
**Status:** Offen · **Phase:** 8 · **Priorität:** Mittel · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** I18N-003, I18N-010
ComparePage, useCompare-Toasts, charts-Namespace (Empty-States, Achsen, Tooltips, TIME_RANGE_OPTIONS-Labels). Paritätstest: Chart-Datenreihen DE == EN (nur Labels/Formatierung verschieden).

### [I18N-015] Account + Billing + Support
**Status:** Offen · **Phase:** 9 · **Priorität:** Hoch · **Aufwand:** L · **Risiko:** Niedrig · **Abhängigkeiten:** I18N-005
AccountPage-Resttexte, getPlanLabel/getBillingStatusLabel/getBillingIntervalLabel → t, CancelSubscriptionModal, BillingPage, Success/Cancel, SupportPage/-Form. Manuell: Stripe-Checkout-Flow DE (unverändert) + EN.

### [I18N-016] Landing + Pricing
**Status:** Offen · **Phase:** 10 · **Priorität:** Mittel · **Aufwand:** L · **Risiko:** Niedrig · **Abhängigkeiten:** I18N-006, I18N-007
LandingPage-Arrays/Props, Landing-Komponenten, IntroSlogan, PricingPage + pricingPlans-Anzeige-Texte, usePageMeta. EN-Marketing-Copy → Betreiber-Review (§ 21.1).

### [I18N-017] Errors, Validation, Toasts, Tour
**Status:** Offen · **Phase:** 11 · **Priorität:** Hoch · **Aufwand:** L · **Risiko:** Mittel · **Abhängigkeiten:** I18N-007
apiErrorMessages.ts (code+String-Match+Passthrough-Kaskade), jobErrors, client.ts-Fallback, ~53 showToast-Stellen, restliche Validierungen, tourSteps/useAppTour. Abschluss: Grep-Sweep (Umlaute + häufige deutsche Wörter) über frontend/src → Restliste bewerten.

### [I18N-018] SEO, a11y, Metadata
**Status:** Offen · **Phase:** 12 · **Priorität:** Mittel · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** I18N-016
usePageMeta auf allen Routen, restliche aria-label/placeholder/title, `<html lang>`-Verifikation, Screenreader-Stichprobe.

### [I18N-019] Paritäts- und Vollständigkeitstests (Querschnitt)
**Status:** Offen · **Phase:** 14 · **Priorität:** Kritisch · **Aufwand:** M · **Risiko:** Niedrig · **Abhängigkeiten:** alle
completeness.test.ts, DE/EN-Request-Paritätstests (Login, Analyse, Definition-Save, Profil-PATCH), Chart-Datenwert-Test, Custom-Definition-Bytegleichheit, Matrizen §§ 25–28 abarbeiten.

## 24. Abhängigkeitsmatrix

| Aufgabe | hängt ab von |
|---|---|
| I18N-001 | — |
| I18N-002 | — |
| I18N-003 | 001, 002 |
| I18N-004 | — (parallel möglich) |
| I18N-005 | 002, 004 |
| I18N-006 | 002 |
| I18N-007 | 002 |
| I18N-008 | 007 |
| I18N-009 | 003, 007 |
| I18N-010 | 003 |
| I18N-011 | 010 |
| I18N-012 | 010, 011 |
| I18N-013 | 010 |
| I18N-014 | 003, 010 |
| I18N-015 | 005 |
| I18N-016 | 006, 007 |
| I18N-017 | 007 |
| I18N-018 | 016 |
| I18N-019 | alle |
| UI-001 | — (unabhängig, vorgezogen) |

## 25. Deutsch-Englisch Functional Parity Matrix

| Funktion | DE | EN | Muss identisch sein | Nachweis |
|---|---|---|---|---|
| Login | ✓ | ✓ | Funktion (Request-Body) | Paritätstest I18N-008 |
| Registrierung | ✓ | ✓ | Funktion | Paritätstest |
| Analyse | ✓ | ✓ | Ergebnis (Rohdaten, Scores, Mode-IDs) | Fixture-Tests I18N-011/012 |
| Compare | ✓ | ✓ | Ergebnis (Datenreihen) | I18N-014 |
| Custom Analysis | ✓ | ✓ | Logik (persistierte Definition byte-identisch) | I18N-013 |
| Favoriten | ✓ | ✓ | Daten | manuell + Request-Log |
| Charts | ✓ | ✓ | Werte (nur Labels/Format verschieden) | I18N-014 |
| Billing | ✓ | ✓ | Subscription (Stripe-Params) | I18N-015 |
| Account | ✓ | ✓ | Nutzerdaten (PATCH-Payload) | I18N-005 |

## 26. Responsive Test Matrix

Pro migrierter Phase: Desktop (1280+), Laptop (~1100), Tablet (768), iPhone (375), Android (~412) × DE × EN. Fokus: Buttons/CTAs (EN teils länger, DE-Komposita teils länger), Sidebar + Mobile-Nav, Tabs/Segmented Controls, Tabellen (horizontal scrollbar im Container), Tooltips, Hero-Headlines, Formulare, Chart-Legenden. Verboten: abgeschnittene/überlappende Texte, horizontales Seiten-Scrollen, springende Buttons, Touch-Targets < 40 px, verschobene Charts. Keine Sprach-Sonderlayouts — Komponenten müssen beide Textlängen tragen (flexWrap/min-width statt fixer Breiten).

## 27. Translation Completeness Matrix

| Namespace | DE | EN | Prüfung |
|---|---|---|---|
| common, nav, auth, landing, pricing, dashboard, analysis, customAnalysis, compare, account, billing, support, errors, validation, metrics, charts, tour, agent | Quelle (verbatim aus Code) | typisiert gegen DE-Schema | TS-Compile + completeness.test.ts (Key-Mengen + keine leeren Strings) |
| legal, admin | deutsch (v1) | — (bewusst nicht übersetzt) | dokumentierte Ausnahme |

## 28. Regressionstest-Matrix

Seiten × Sprachen (DE, EN) × Zustände: LandingPage, Login, Registrierung, Forgot/Reset/Verify, Dashboard, AnalysisPage, CustomAnalysis (im Analyze), ComparePage, AccountPage, BillingPage (+Success/Cancel), Support, Navigation/Sidebar/Mobile-Nav, Legal (UI-Chrome), Cookie-Banner. Sprachwechsel-Matrix: DE→EN und EN→DE jeweils ausgeloggt, eingeloggt, auf Dashboard/Analyze/Compare/Account/Billing, nach Reload, neuer Tab, neues Gerät/Session, Logout+Login. Prüfpunkte je Zelle: Sprache korrekt, Preference gespeichert, keine Daten verloren, Navigation intakt, Analyse unverändert, keine unnötigen Requests, keine UI-Fehler.

## 29. Migration und Rollback

- Jede Phase = eigene Commits; Revert einer Phase lässt alle anderen funktionsfähig (Namespaces sind additiv).
- DB: `alembic downgrade` entfernt `users.locale` verlustfrei; Frontend toleriert fehlendes Feld (optionales Property).
- Kill-Switch UI: `SUPPORTED_LOCALES = ["de"]` deaktiviert EN app-weit (Switcher verschwindet, detect liefert immer "de"), ohne Code-Rückbau.
- EN-Lazy-Chunk fehlgeschlagen → `ready` bleibt false → DE bleibt sichtbar (kein Fehlerzustand).

## 30. Adding a New Language Checklist

1. Locale in `SUPPORTED_LOCALES` + `INTL_LOCALES` registrieren (config.ts)
2. `locales/<xx>/` anlegen: alle Namespaces, typisiert gegen DE-Schema (Compiler erzwingt Vollständigkeit)
3. Formatierungsregeln prüfen (format.ts: Kompakt-Suffixe der neuen Sprache ergänzen; `Intl.*` übernimmt Zahlen/Datum automatisch)
4. Server-Validierung erweitern (erlaubte locale-Werte, api/routes/auth.py)
5. Switcher-UI: ab ≥4 Sprachen Segmented Control → Dropdown
6. completeness.test + Responsive-Stichprobe + `npm run build`
Nicht nötig: Komponenten-Umbau, Business-Logik, DB-Änderung (Spalte generisch), Routing.

## 31. Definition of Done

- Architektur: i18n-Schicht vollständig, DE/EN produktiv, weitere Sprachen per Checkliste § 30 ergänzbar
- Strings: alle Bereiche aus § 5 migriert (außer dokumentierte Ausnahmen legal/admin); dynamische Texte, Fehler, Validation, a11y abgedeckt
- Business Logic: alle Paritätstests grün; Legacy-Key eingefroren; Custom-Definitionen byte-stabil; API-Parameter/Cache-Keys locale-frei
- Account: Spracheinstellung mit Persistenz, Bestandsnutzer (NULL) unverändert, ausgeloggte Nutzer per detect
- Localization: Zahlen/Prozent/Datum locale-aware, Währung strikt getrennt
- UX: alle Hauptseiten beide Sprachen responsive ohne Layoutbrüche; Sprachwechsel zerstört keinen Zustand
- SEO: Laufzeit-Title/Meta pro Sprache; hreflang-Verzicht dokumentiert
- Testing: Characterization-, Paritäts-, Completeness-, Missing-Key-, Responsive-Tests vorhanden und grün
- AccountPage-Bug: behoben (UI-001 ✓)

## 32. Final Verification Checklist

- [ ] `npm run build` grün (echter Build) + vitest vollständig grün + pytest vollständig grün
- [ ] DE-Durchklick aller Seiten: pixelnah identisch zu Vorher-Screenshots
- [ ] EN-Durchklick aller Seiten: vollständig übersetzt (Grep-Sweep leer), keine Layoutbrüche
- [ ] Sprachwechsel-Matrix § 28 vollständig
- [ ] Network-Nachweis: identische Requests unter DE und EN (Login, Analyse, Save, PATCH)
- [ ] Laufende Analyse überlebt Sprachwechsel (Job-Ergebnis identisch)
- [ ] Custom-Definition vor/nach Wechsel byte-identisch (DB-Vergleich)
- [ ] Bestandsnutzer (locale=NULL) verhält sich exakt wie vor dem Projekt
- [ ] Migration up/down auf DB-Kopie verifiziert
- [ ] Glossar-HTML byte-identisch
- [ ] Entry-Chunk-Budget nachgemessen (§ 19)
- [ ] EN-Copy fachlich reviewt (Betreiber)

---

# AccountPage Profil Spacing Bug (UI-001)

**Status:** Umgesetzt 2026-07-18 · **Priorität:** Hoch · **Aufwand:** XS · **Risiko:** Niedrig · **Abhängigkeiten:** keine

## Beobachtung
Im Bereich „Profil" der AccountPage stehen die Buttons („Bearbeiten", „Tour erneut starten") direkt unter der letzten Info-Zeile („Geburtsdatum") ohne vertikalen Abstand.

## Ursache (verifiziert)
`frontend/src/pages/app/AccountPage.tsx`, Profil-Sektion (`data-tour="account-profile-section"`). Im **Nicht-Editiermodus** liegen `<div style={infoList}>` (flex column, `gap:14px`, kein marginBottom) und `<div style={passwordActionRow}>` (kein marginTop) als direkte Geschwister in einem Fragment ohne Wrapper-gap → **0 px Abstand**. Der Editiermodus ist korrekt (`passwordForm` mit `gap:22px`). `passwordActionRow` wird an 3 Stellen genutzt; nur die Instanz nach `infoList` (~Z. 797) ist fehlerhaft. Vergleichsmuster im selben File: `membershipInfoList` hat `marginBottom:18px`, `heroActionRow` hat `marginTop:26px`.

## Fix (kleinstmöglich)
Nur an der fehlerhaften Instanz: `style={{ ...passwordActionRow, marginTop: "24px" }}`. Geteilte Style-Objekte (`infoList`, `passwordActionRow`) und die beiden korrekten Instanzen bleiben unangetastet. Kein Redesign, keine strukturelle Änderung.

## Tests
- Desktop + Mobile: Abstand vorhanden, kein Umbruchfehler (Buttons haben `flexWrap:"wrap"`)
- Editiermodus unverändert (eigene Form-Instanz)
- Nach i18n-Phase 9: erneute Prüfung mit englischen (teils längeren) Buttontexten

## Rollback
Einzeiler revertieren.
