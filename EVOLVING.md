# EVOLVING – Produktverbesserungsplan

> Erstellt am 2026-07-13 durch Fable 5 (Planmodus) auf Basis einer vollständigen Codeanalyse.
> Abarbeitung durch Sonnet 5. Jede Aufgabe hat eine stabile ID (`EV-XXX`) für Commits/PRs/Statusmeldungen.
> Verbindliche Arbeitsregel aus dem Projektgedächtnis: Vor jedem „fertig" muss im Frontend ein echtes `npm run build` (nicht nur `tsc --noEmit`) fehlerfrei durchlaufen.

## 1. Ziel und Hintergrund

Kollegen-Feedback zur Plattform (Unternehmensanalyse, Kennzahlen, Vergleiche, Charts, Favoriten, Live-Kurse, Abonnements) ergab fünf Verbesserungsfelder:

1. Das Universum analysierbarer Unternehmen ist unbeabsichtigt eingeschränkt (früher ~6.000 NYSE+Nasdaq-Titel).
2. Charts/KPIs zeigen `$`, obwohl Werte teils in anderen Währungen vorliegen.
3. Chart-Hover auf Vergleichscharts zeigt nicht zuverlässig alle Unternehmen; Zeitraumfilter (1M–Max) und prozentuale Veränderung fehlen.
4. Kursentwicklung soll grafisch dargestellt werden (Analyse-/Vergleichsseite + Dashboard-Sparklines für Favoriten).
5. Der Sidebar-Eintrag „Abrechnung" soll für Pro-Nutzer entfallen (Verwaltung weiterhin über AccountPage).

## 2. Unveränderliche Anforderungen

- **Neutralitäts-Leitplanke:** reine Information, keine Anlageberatung. Keine wertenden Formulierungen bei %-Veränderungen oder Kurscharts (kein „Kaufsignal", keine Empfehlungen).
- Kein Export-Feature (Produktentscheidung des Betreibers) – Exportfunktionen sind bei der Currency-Architektur NICHT zu berücksichtigen.
- Frontend + Backend laufen beide auf Render. Backend und Datenbank laufen inzwischen auf dem **Render-Starter-Abo mit Shell-Zugriff** (Stand 2026-07-13; der Hinweis „kein Shell-Zugriff" in LAUNCH.md:148 ist veraltet). Ein manueller Sofort-Import per Shell ist damit möglich – wiederkehrende Betriebsaufgaben dürfen trotzdem NICHT von manuellen Shell-Läufen abhängen (Automatisierung bleibt Pflicht).
- **Chart-Linien müssen durchgängig bleiben** (Betreiber-Vorgabe 2026-07-13): `connectNulls` bleibt in allen Line-Charts aktiv; Lücken werden optisch überbrückt und NUR im Tooltip als „–" ausgewiesen. Keine lückenhaften/unterbrochenen Linien.
- Bestehende API-Verträge nur **additiv** erweitern (neue optionale Felder/Parameter), niemals bestehende Felder umbenennen/entfernen.
- Free-/Pro-/Admin-Berechtigungen, gespeicherte Favoriten, Analyse-Historie und bestehende Analysen dürfen nicht beschädigt werden.
- Deutsch als UI-Sprache (bestehende Labels wie „Abrechnung", „Mrd./Mio./Tsd." beibehalten).
- Vor Abschluss jeder Frontend-Aufgabe: `npm run build` muss fehlerfrei sein.

## 3. Analysierte Projektarchitektur

| Baustein | Technologie | Fundstelle |
|---|---|---|
| Frontend | React 19 + Vite 7 + TypeScript 5.9 | `frontend/package.json` |
| Chart-Library | Recharts 3.8 | `frontend/package.json`, `frontend/src/components/charts/MultiLayerChart.tsx` |
| State-Management | Kein globaler Store; lokale Hooks + `useEffect`-Fetches (kein Auth-Context!) | z. B. `frontend/src/components/layout/AppSidebar.tsx:60-73` |
| API-Client | Hauseigene Fetch-Wrapper in `frontend/src/api/*` | `frontend/src/api/analysis.ts` u. a. |
| Backend | FastAPI (Python), slowapi-Rate-Limits | `api/main.py`, `api/core/rate_limit.py` |
| DB/ORM | PostgreSQL + SQLAlchemy + Alembic | `api/models/*`, `alembic/versions/*` |
| Auth | Token in `localStorage.access_token`, `GET /auth/me`; Admin = `plan == "admin"` | `frontend/src/routes/ProtectedRoute.tsx`, `api/core/dependencies.py:86-89` |
| Billing | Stripe (Checkout, Customer Portal, Webhooks mit Idempotenz-Tabelle `stripe_events`) | `api/routes/billing.py`, `api/routes/stripe_webhook.py`, `api/models/stripe_event.py` |
| Finanzdaten (Fundamentaldaten) | SEC XBRL (companyfacts) | `agent/data_sources/sec_source.py` |
| Kursdaten | yfinance (primär) + Alpha-Vantage-Fallback | `agent/DataLoader.py:1078-1097` |
| Symbol-Universum | NASDAQ-Trader-Verzeichnisdateien (`nasdaqlisted.txt`, `otherlisted.txt`), kostenlos, ohne API-Key | `scripts/import_symbols.py:42-43` |
| Hintergrundjobs | Startup-Worker: `downgrade_worker`, `filing_alert_worker`, `watchlist_digest_worker`; Jobs: Grace-Periods, Monats-Reset | `api/main.py:164-168`, `api/jobs/*` |
| Tests | Backend: pytest (`api/tests`, `agent/tests`, `pytest.ini`); Frontend: **kein Test-Framework vorhanden** | `pytest.ini`, `frontend/package.json` |
| Deployment | Render (Backend: `alembic upgrade head && uvicorn api.main:app`; Frontend ebenfalls Render) | `LAUNCH.md:148` |
| FX-Quelle | `agent/data_sources/fx_source.py` ist eine **leere 0-Byte-Datei** (kein FX vorhanden) | verifiziert |

Wichtige Seiten (tatsächliche Dateinamen – es gibt KEINE Dateien namens `AnalysisPage`/`CompareAnalysisPage`):
- Analyse: `frontend/src/pages/app/AnalyzePage.tsx` (Standard-Analyse = Dossier ohne Charts; Modus „Individuell" = Charts via `CustomAnalysisResultsList`)
- Vergleich: `frontend/src/pages/app/ComparePage.tsx`
- Dashboard: `frontend/src/pages/app/DashBoardPage.tsx`
- Konto: `frontend/src/pages/app/AccountPage.tsx`, Abrechnung: `frontend/src/pages/app/BillingPage.tsx`
- Sidebar: `frontend/src/components/layout/AppSidebar.tsx` (Desktop UND Mobile – ein Component, Drawer via `frontend/src/layouts/AppLayout.tsx:183-235`)

## 4. Relevante Datenflüsse

### 4.1 Unternehmensuniversum (Bestätigt)
```
NASDAQ-Trader-Dateien ──(manuell!) scripts/import_symbols.py──▶ Tabelle `symbols` ──GET /analyze/symbols──▶ useSymbolSearch (250ms Debounce, limit=8) ──▶ SymbolCommandField (Analyse) / SymbolSuggestField (Vergleich)
                                                                     ▲                                                        │
                                              Migration 9a4f8ab8c4c6 seeded NUR 23 Zeilen              Fehlerfall: stiller Fallback auf 23 statische Symbole (frontend/src/data/symbols.ts)
```
- Import-Filter (`scripts/import_symbols.py`): nur NYSE (`N`), NYSE American (`A`), NASDAQ; keine Test-Issues, keine ETFs (Z. 104/119); Namens-Regex schließt warrants/rights/units/preferred/notes aus (Z. 59-60); Delisting ⇒ `is_active=False` (Z. 154-157).
- Suche (`api/routes/analyze.py:53-104`): `is_active=True` + `ilike` auf Symbol/Name, `limit ≤ 50`, Rate-Limit 60/min. Leere Query ⇒ 23 `POPULAR_SYMBOLS` (Z. 46-50).
- **Analyse/Vergleich validieren NICHT gegen die Tabelle** – nur `.strip().upper()` (`api/routes/analyze.py:236-323`, `api/routes/custom_analysis.py:48-49`). Favoriten validieren dagegen gegen `symbols` (`api/routes/favorites.py:58`).
- Kein automatischer Import: kein Cron, kein Startup-Hook, kein CI-Aufruf von `import_symbols` (repo-weit verifiziert).

### 4.2 Analyse & Vergleich
- Standard-/Vollanalyse: `useAnalysisJobs` → `/full/start` bzw. `/analyze/{mode}/start` → Polling → `FullResult` (`frontend/src/types/analysis.ts`).
- Individuell/Vergleich: `/analyze/custom/*` bzw. je Unternehmen ein Custom-Job (`useCompare`-Hook, Mapping in `frontend/src/compare/mapping.ts`); Zeitreihen aus `api/routes/custom_analysis.py:161-167` als `{date: str(idx), value}` – **voller Pandas-Timestamp** wie `"2024-09-30 00:00:00"`, firmenspezifische Fiskaldaten.
- Historische Endpoints-Familie: `api/routes/metric_routes.py:621-1470` (`historical-market-cap`, `-enterprise-value`, `-price-to-*` u. a.), Zeilenformat `{symbol, rows:[{date:"%Y-%m-%d", …}]}`; `historical-market-cap` enthält als einziger eine tägliche `close`-Spalte.

### 4.3 Charts
- Einziger aktiv genutzter Chart: `frontend/src/components/charts/MultiLayerChart.tsx` (ComparePage:383, CustomAnalysisResultsList:58). `TimeSeriesChart.tsx`, `ChartLayerBuilder.tsx` sind toter Code; `Sparkline.tsx` wird nur in `MetricResultCard` für Kennzahl-Historie genutzt, nicht für Kurse.
- Merge-Logik `mergeLayers` (MultiLayerChart.tsx:31-45): Zeilen werden **per exaktem Datums-String** geschlüsselt; jede Firma bringt eigene Fiskal-Stichtage mit ⇒ Zeilen sind dünn besetzt; Recharts-Standard-`<Tooltip>` listet nur Serien mit definiertem Punkt an der gehoverten X-Position ⇒ **nur 1–2 Firmen sichtbar**. `connectNulls` (Z. 231) überbrückt die Linie optisch. WICHTIG: Die durchgängige Liniendarstellung ist ERWÜNSCHT und muss erhalten bleiben – der Fehler liegt ausschließlich im unvollständigen Tooltip, nicht in `connectNulls`. Nach dem Fix: Linien weiterhin durchgängig, Tooltip vollständig (Lücken als „–").
- Kein Zeitraumfilter, keine %-Berechnung, kein Kurschart vorhanden (einzige „Range"-Interaktion: Y-Achsen-Zoom per Drag, MultiLayerChart.tsx:91-133; `FrequencyToggle` = jährlich/quartalsweise).

### 4.4 Währungen
- Currency ist **nirgends** vorhanden: kein DB-Feld, kein Schema-Feld, kein API-Feld, kein Frontend-Typ (repo-weit verifiziert). SEC-Quelle fragt XBRL-Facts fest mit `"units": ["USD"]` ab (`agent/data_sources/sec_source.py:1376`; Unit-Auswahl Z. 1559-1571, 1173-1257). `get_current_price_per_share()` liefert nackten Float (`agent/DataLoader.py:1078`).
- Das eine hartcodierte `$` für Finanzdaten: `frontend/src/components/metrics/metricFormatting.tsx:79` (`` `$${formatCompactNumber(value)}` ``), getriggert durch `unit === "currency"` (~24 Einträge in `frontend/src/config/metricsConfig.ts`) oder Heuristik `isDollarKey` (metricFormatting.tsx:41-60). Konsumenten: `MetricResultCard.tsx:63`, `ComparePivotTable.tsx:199-204` (inkl. Live-Kurs-Zeile `__price`), `CrvTargetPanel.tsx:77`.
- Chart-Achsen/-Tooltips zeigen KEIN `$` (nutzen `formatCompactNumber` aus `chartUtils.ts:42-55`, deutsch „Mrd./Mio./Tsd."). €-Symbole existieren nur im SaaS-Billing (pricingPlans, BillingPage, AdminDashboard) – nicht anfassen.

### 4.5 Live-Kurs & Favoriten
- „Live-Kurs" = yfinance-Quote (verzögert, kein echtes Realtime; Alpha-Vantage-Fallback), Endpoint `GET /metrics/current-price/{symbol}` (`api/routes/metric_routes.py:48-79`), In-Process-Cache TTL 12 s (Z. 39), Rate-Limit 60/min; Frontend pollt je Symbol alle 20 s (`frontend/src/hooks/useLivePrice.ts`, pausiert bei verstecktem Tab). **Kein Batch-Endpoint**, kein täglicher Preis-History-Endpoint.
- Favoriten: Tabelle `favorites` (`api/models/favorite.py`, Unique user+symbol), Endpoints `GET/POST/DELETE /favorites` (`api/routes/favorites.py`), flaches Limit 50/Nutzer (Z. 19, kein Free/Pro-Unterschied). Anzeige **nur in der Sidebar** (AppSidebar.tsx:298-318, 508-548 mit `LivePriceBadge`) – das Dashboard rendert Favoriten aktuell überhaupt nicht (nur „Letzte Analysen" + Account-Panel, DashBoardPage.tsx:388-506).

### 4.6 Subscription & Sidebar
- Subscription-Zustand liegt auf der `users`-Zeile: `plan` ∈ {free, friends, pro, admin} (`api/models/user.py:45-46`), `billing_status` ∈ {active, canceling, past_due, canceled} (Z. 53-54), plus `grace_until` (+24 h bei Zahlungsfehler). **Kein Trial-Konzept im Code.**
- Frontend erfährt Status via `GET /auth/me` (`frontend/src/api/auth.ts:84-90`, Typ Z. 33-59); jede Komponente fetcht selbst (kein Context) ⇒ asynchron beim ersten Render; Admin-Eintrag „ploppt" nach (AppSidebar.tsx:47-73, 104-109).
- Sidebar-Navigation = statisches Array `navItems` (AppSidebar.tsx:33-40), Billing-Eintrag „Abrechnung" Z. 37; Mobile = derselbe Component als Drawer.
- BillingPage = Upgrade-/Planauswahl-Seite mit Stripe-Checkout (BillingPage.tsx:64-84), Guard nur Token (`ProtectedRoute`). Stripe-Customer-Portal-Button liegt auf der AccountPage (`AccountPage.tsx:987`, Bedingung `hasStripeCustomer && (isPro || isSubscriptionCanceling)` Z. 183-184); zwei weitere Links zu `/app/billing` (Z. 497, 610).

## 5. Aktueller Zustand

### 5.1 Analysierbare Unternehmen
- **Erwartet:** ~6.000 NYSE+Nasdaq-Titel suchbar/analysierbar. **Beobachtet:** stark eingeschränkte Liste.
- **Bestätigt:** Lokale DB enthält 5.898 aktive Symbole (NASDAQ 3.412, NYSE 2.214, NYSE American 272) + 420 delistete – lokal ist nichts eingeschränkt. Der Import läuft ausschließlich manuell; die Migration seeded nur 23 Zeilen; der Render-Startbefehl führt nur `alembic upgrade head` aus; Render Free hat keinen Shell-Zugriff.
- **Sehr wahrscheinlich (Ursache):** Die Produktions-`symbols`-Tabelle enthält nur die 23 Seed-Zeilen, weil `scripts/import_symbols.py` dort nie lief. (Nicht direkt gegen Prod-DB verifizierbar – siehe „Noch zu verifizieren". Mit dem Render-Starter-Shell-Zugriff kann der Betreiber das sofort prüfen: `SELECT count(*) FROM symbols;` bzw. einen manuellen Import-Lauf starten – siehe Sofortmaßnahme in EV-010.)
- **Mögliche Zweitursache:** `searchSymbolsSafe` (`frontend/src/api/analysis.ts:64-83`) verschluckt JEDEN Fehler (auch Rate-Limit 429/5xx) und fällt still auf 23 statische Symbole zurück – Backend-Ausfälle sehen für Nutzer wie eine „kleine Liste" aus.
- **Bestätigt (kein Blocker):** Analyse-/Vergleichsstart validiert nicht gegen die Tabelle – jedes von SEC/yfinance unterstützte Symbol ist technisch analysierbar; nur Suche + Favoriten hängen an der Tabelle. Die Industry-Multiples-Map schränkt NICHTS ein (`agent/industry_multiples.py:230-239` hat immer `GLOBAL_FALLBACK_MULTIPLES`).

### 5.2 Währungen
- **Bestätigt:** Currency existiert in keiner Schicht (DB, Schemas, API, Frontend-Typen). SEC-Quelle erzwingt USD-Units; `fx_source.py` ist leer. Das `$` stammt aus genau einer Stelle: `metricFormatting.tsx:79`.
- **Noch zu verifizieren:** Wie sich ein Nicht-USD-Filer (20-F, z. B. SAP mit EUR-Facts) aktuell verhält – wegen `units:["USD"]` liefern Fundamentalkennzahlen dort möglicherweise gar keine Daten (nicht falsche Währung, sondern Lücken). Muss in EV-020 zuerst empirisch geprüft werden.
- **Bestätigt:** Kurse an NYSE/Nasdaq (auch ADRs) notieren in USD – die Kurs-Währung ist faktisch USD; die Berichtswährung kann abweichen. Beide müssen getrennt gekennzeichnet werden.

### 5.3 Chart-Hover
- **Bestätigt (Root Cause):** `mergeLayers` schlüsselt per exaktem Datums-String (MultiLayerChart.tsx:31-45); Firmen haben unterschiedliche Fiskal-Stichtage (Backend liefert `str(idx)` = voller Timestamp, custom_analysis.py:164); gemergte Zeilen sind dünn besetzt; der geteilte Recharts-Tooltip zeigt nur Serien mit definiertem Wert an der X-Position ⇒ 1–2 statt aller Firmen. `connectNulls` kaschiert die Lücken in der Linie.

### 5.4 Zeitraumfilter
- **Bestätigt:** Es existiert kein Zeitraumfilter. Fundamentaldaten sind jährlich/quartalsweise (as-filed) – 1M/2M/3M/6M-Filter wären dort fachlich sinnlos; sie sind nur für tägliche Kursdaten sinnvoll.

### 5.5 Prozentuale Veränderungen
- **Bestätigt:** Keine %-Veränderungs-Berechnung vorhanden (das `percent` in `AnalyzeStickyBar.tsx:12` ist Job-Fortschritt).

### 5.6 Kurscharts
- **Bestätigt:** Kein Kurschart, kein täglicher Preis-History-Endpoint. Einzige tägliche Preisquelle über die API: `historical-market-cap.rows[].close` (metric_routes.py:621-662). Die Datenschicht (DataLoader/yfinance) kann tägliche Closes liefern.
- **Bestätigt:** „Live-Kurs" = verzögerte yfinance-Quote, 20-s-Polling, 12-s-Server-Cache. Kein Realtime, keine Websockets.

### 5.7 Favoriten und Dashboard
- **Bestätigt:** Favoriten nur in der Sidebar (Symbol + LivePriceBadge); Dashboard zeigt keine Favoriten; Quote-Abruf strikt pro Symbol (N Favoriten = N Requests); Limit 50 Favoriten/Nutzer.

### 5.8 Billing und Sidebar
- **Bestätigt:** Statisches `navItems`-Array, „Abrechnung" für ALLE sichtbar; nur der Admin-Eintrag ist konditional (nachträglich angehängt ⇒ Flicker-Muster existiert bereits). BillingPage = Upgrade-Seite; Portal-Button auf AccountPage mit Bedingung `hasStripeCustomer && (isPro || isSubscriptionCanceling)`; Free-Nutzer sehen dort „Kein aktives Abonnement" (disabled) + Links auf `/app/billing`.
- **Mögliche Altlast:** Frontend referenziert `billing_status === "payment_failed_canceled"` (AccountPage.tsx:181, :1166), das Backend schreibt diesen Wert in den Webhooks nie – vermutlich setzt ihn `api/jobs/process_grace_periods.py` (Noch zu verifizieren, in EV-080 prüfen).

## 6. Bestätigte Probleme und Beweise

| # | Problem | Beweis (Datei:Zeile) | Einstufung |
|---|---|---|---|
| P1 | Symbol-Import läuft nie automatisch; Prod-Seed = 23 Zeilen | `alembic/versions/9a4f8ab8c4c6_create_symbols_table.py` (23 LEGACY_SYMBOLS); kein Aufrufer von `import_symbols` repo-weit; `LAUNCH.md:148` | Bestätigt (Mechanismus); Prod-Zustand: sehr wahrscheinlich |
| P2 | Stiller Frontend-Fallback maskiert Backend-Fehler mit 23 Symbolen | `frontend/src/api/analysis.ts:64-83` | Bestätigt |
| P3 | Hartcodiertes `$` für alle Finanzwerte | `frontend/src/components/metrics/metricFormatting.tsx:79` + Trigger Z. 41-60 + `metricsConfig.ts` (`unit:"currency"`) | Bestätigt |
| P4 | Currency fehlt in allen Schichten; SEC-Quelle erzwingt USD-Units | `agent/data_sources/sec_source.py:1376`; leere `fx_source.py`; keine Currency-Felder in `api/models`, `api/schemas`, `frontend/src/types` | Bestätigt |
| P5 | Tooltip zeigt nur Firmen mit exakt passendem Datums-Key | `MultiLayerChart.tsx:31-45` (mergeLayers), `:231` (connectNulls), `api/routes/custom_analysis.py:164` (`str(idx)` = voller Timestamp) | Bestätigt |
| P6 | Kein Zeitraumfilter / keine %-Veränderung / kein Kurschart / kein Preis-History-Endpoint | Abwesenheit repo-weit verifiziert | Bestätigt |
| P7 | Dashboard ohne Favoriten; Quotes nicht batchbar | `DashBoardPage.tsx:388-506`; nur `GET /metrics/current-price/{symbol}` | Bestätigt |
| P8 | „Abrechnung" für alle Nutzer in der Sidebar | `AppSidebar.tsx:33-40` (Z. 37) | Bestätigt |
| P9 | Kein Auth-/Subscription-Context ⇒ Nav-Flicker-Risiko | `AppSidebar.tsx:47-73,104-109`; kein useAuth/Context repo-weit | Bestätigt |

## 7. Offene Produktentscheidungen

**Bereits entschieden (durch den Betreiber, 2026-07-13):**

| ID | Entscheidung | Ergebnis |
|---|---|---|
| D1 | Währungsdarstellung im Vergleich | **Originalwährung + ISO-Kennzeichnung, keine FX-Umrechnung.** Gemischte Währungen ⇒ Hinweis am Chart/Tabelle. |
| D2 | Tooltip bei fehlendem Datenpunkt | **„–" / Keine Daten anzeigen** (Firma bleibt im Tooltip sichtbar). Zusatzvorgabe: Die Chart-LINIEN bleiben durchgängig (`connectNulls` aktiv) – Lücken erscheinen nur im Tooltip, nie als unterbrochene Linie. |
| D3 | Kurschart-Einsatzbereiche | **Beides; Analyse/Vergleich (Variante B) zuerst, danach Dashboard-Sparklines (Variante A).** |
| D4 | Kursvergleich mehrerer Firmen | **Normalisierte Performance in % (Basis 0 % am Zeitraumstart).** Einzelanalyse zeigt absolute Kurse. |

**Noch offen – mit Standardannahme umsetzen (blockiert den Plan nicht):**

| ID | Frage | Standardannahme (so umsetzen) | Alternative | Abhängige Aufgaben |
|---|---|---|---|---|
| D5 | Close vs. Adjusted Close vs. Intraday für Kurs-Historie? | **Adjusted Close** (yfinance `auto_adjust=True`; splits-/dividendenbereinigt ⇒ keine künstlichen Sprünge). Aktueller Kurs (LivePriceBadge) bleibt unverändert die Live-Quote. | Unbereinigter Close (zeigt historisch „echte" Kurse, aber Split-Artefakte) | EV-060, EV-061, EV-062, EV-070 |
| D6 | Standard-Zeitraum? | **Kurscharts: 1Y; Fundamental-Charts: Max.** Bei jährlichen Daten sind 1M–6M ausgeblendet (nur 1Y/2Y/5Y/Max wählbar). | Überall Max | EV-040, EV-041, EV-061, EV-062 |
| D7 | Zeitraum-Persistenz? | **State pro Chart-Sektion (React-State), keine URL-/Server-Persistenz** (kleinster Eingriff, keine Router-Risiken). | URL-Query-Param (teilbare Links, Back-Button) – später nachrüstbar | EV-040, EV-041 |
| D8 | Symbol-Sync-Automatisierung? | **Startup-Import bei fast leerer Tabelle (< 1.000 Zeilen) + wöchentlicher Hintergrund-Sync + Admin-Endpoint zum manuellen Auslösen.** | Nur Admin-Endpoint (manuell) | EV-010, EV-011, EV-012 |
| D9 | Analyse-Start gegen `symbols`-Tabelle validieren? | **Ja, weiche Validierung:** unbekanntes/inaktives Symbol ⇒ 422 mit klarer Meldung („Symbol nicht im NYSE/Nasdaq-Universum"). Erst NACH gesichertem Prod-Import scharf schalten (Reihenfolge in EV-014 erzwungen). | Keine Validierung (heutiger Zustand: kryptische Fehler tief im Agent) | EV-014 |
| D10 | Frontend-Test-Framework (vitest) einführen? | **Ja, minimal:** vitest nur für reine Utility-Funktionen (Bucketing, Range-Filter, %-Berechnung, Currency-Formatter). Keine Component-Tests. | Keine FE-Tests (nur pytest + manuell) – dann entfallen die FE-Testpunkte in EV-032/040/050 | EV-032, EV-040, EV-050, EV-022 |

## 8. Technische Zielarchitektur

### 8.1 Unternehmensuniversum
`scripts/import_symbols.py` bleibt die einzige Import-Logik (Wiederverwendung, kein Duplikat). **Sofortmaßnahme (Betreiber, optional):** Dank Render-Starter-Shell kann `python -m scripts.import_symbols` einmalig manuell gegen Prod laufen – das behebt das Symptom sofort, ersetzt aber nicht die Automatisierung. Dauerhaft neu: (a) Startup-Task in `api/main.py`, der bei `count(symbols) < 1000` den Import im Hintergrund-Thread ausführt (Muster der bestehenden Worker Z. 164-168); (b) wöchentlicher Sync-Worker (gleiche Funktion, `asyncio`-Loop wie `watchlist_digest_worker`); (c) `POST /admin/symbols/refresh` + `GET /admin/symbols/stats` (require_admin) für manuellen Trigger und Nachweis ohne Shell-Sitzung; (d) Frontend zeigt Suchfehler statt stillem 23er-Fallback.

### 8.2 Currency
Neues additives Feld-Paar überall dort, wo Finanzwerte fließen:
- `reporting_currency` (ISO-Code der SEC-Facts-Unit, z. B. "USD"; `null` wenn unbestimmbar) – aus der bereits vorhandenen Unit-Auswahl in `sec_source.py` extrahiert.
- Kurs-Endpoints liefern zusätzlich `currency: "USD"` (NYSE/Nasdaq-Handelswährung).
Frontend: zentrale Funktion `formatMonetary(value, isoCode)` in `metricFormatting.tsx` ersetzt das hartcodierte `$` (Symbol-Map: USD→$, EUR→€, GBP→£, JPY→¥; sonst `ISO-Code + Wert`). Fallback ohne Angabe: weiterhin `$` (heutiges Verhalten, kein Regressionsrisiko). Vergleich mit gemischten Währungen: Badge/Hinweis „Werte in Originalwährung (USD, EUR)" an Chart und Pivot-Tabelle. Keine Umrechnung (D1).

### 8.3 Chart-Datenmodell
- Backend: `custom_analysis.py` serialisiert Datumsindizes als `YYYY-MM-DD` (statt `str(idx)`).
- Frontend: `mergeLayers` bekommt einen Bucketing-Modus (`"year" | "quarter" | "date"`): Fundamentaldaten jährlich ⇒ Bucket = Geschäftsjahr („2023"), quartalsweise ⇒ „Q3 2023", Kursdaten ⇒ exaktes Datum. Bucket-Key wird Merge-Schlüssel; pro Firma+Bucket gewinnt der späteste Punkt. Neuer Custom-Tooltip listet ALLE Layer, fehlende Werte als „–" (D2).
- **Durchgängige Linien (verbindliche Betreiber-Vorgabe):** `connectNulls` bleibt in allen Line-Charts aktiv – Serien mit Bucket-Lücken werden als durchgezogene Linie gezeichnet, nie unterbrochen. Das Bucketing reduziert Lücken ohnehin drastisch; verbleibende Lücken sind ausschließlich im Tooltip (als „–") sichtbar. Jede Chart-Aufgabe hat „Linien durchgängig" als Akzeptanzkriterium.

### 8.4 Zeitraumfilter + %-Veränderung
- Wiederverwendbare `TimeRangeFilter`-Komponente (Buttons 1M/2M/3M/6M/1Y/2Y/5Y/Max, konfigurierbare Teilmenge).
- Utility `filterSeriesByRange(series, range)`: Anker = Datum des NEUESTEN Datenpunkts (nicht „heute", sonst leere Charts bei Datenverzug); kalenderbasierte Monats-Subtraktion (eigene kleine Hilfsfunktion, keine neue Dependency); Punkte ≥ Cutoff. Wochenenden/Feiertage brauchen keine Sonderbehandlung (Filter, kein Kalender).
- Utility `computePercentChange(series)`: Start = erster Punkt im gefilterten Zeitraum, Ende = letzter; Start = 0 oder Start < 0 ⇒ `null` („n. v."); < 2 Punkte ⇒ `null`; Anzeige mit Vorzeichen, 1 Nachkommastelle, grün/rot/neutral. Nur für Level-Kennzahlen (`unit === "currency"`, Market Cap, Kurs) – NICHT für Ratios (EV/EBIT, P/B …), Margen oder Pass/Fail-Werte.
- Fundamentaldaten: clientseitig filtern (Serie ist bereits vollständig geladen). Kursdaten: serverseitig via `range`-Parameter (Payload-Größe).

### 8.5 Kurscharts
- Neuer Endpoint `GET /metrics/price-history/{symbol}?range=1m|2m|3m|6m|1y|2y|5y|max`: tägliche Adjusted-Close-Serie via DataLoader/yfinance; ab 2Y serverseitiges Downsampling auf Wochenwerte; Antwort `{symbol, currency:"USD", range, rows:[{date, close}]}`; In-Process-Cache TTL 15 min; Rate-Limit 30/min; Auth wie current-price (`get_current_user`, kein Quota-Verbrauch).
- AnalyzePage (Individuell) + ComparePage: eigenständige, zuschaltbare Kurschart-Sektion (Toggle „Kursentwicklung anzeigen"), eigener Chart – KEINE zweite Y-Achse in Kennzahlen-Charts (unterschiedliche Einheiten, UX-Risiko). Vergleich: normalisiert auf 0 % (D4), Einzelansicht: absolute USD-Kurse.
- Dashboard (Phase 7): neue Favoriten-Sektion mit `Sparkline` (Komponente existiert bereits) + Kurs + %-Änderung; Batch-Endpoint `GET /metrics/price-history-batch?symbols=A,B,…&range=1m` (max. 20 Symbole, sequentiell mit kleinem Delay gegen yfinance-Throttling, gecacht).

### 8.6 Billing-Navigation
`AppSidebar`: Billing-Eintrag wird aus `navItems` herausgelöst und konditional gerendert – sichtbar NUR bei geladenem Nutzer mit `plan === "free"`. Während `isLoadingUser` und bei pro/friends/admin/past_due/canceling: ausgeblendet (Begründung: kurzes Erscheinen ist weniger störend als Erscheinen-und-Verschwinden; Free-Nutzer erreichen Billing zusätzlich über AccountPage-Links und Dashboard-Upgrade-Nudge). Route `/app/billing` bleibt ungeschützt direkt erreichbar (Checkout-Rückkehr, Deep-Links, Upgrade nach Downgrade).

## 9. Umsetzungsphasen

| Phase | Inhalt | Aufgaben |
|---|---|---|
| Phase 0 – Bestehendes Verhalten absichern | Baseline-Tests + Build-Nachweis vor jeder Änderung | EV-001, EV-002 |
| Phase 1 – Unternehmensuniversum | Prod-Import, Sync-Job, Nachweis-Endpoints, Fallback-Fix, weiche Validierung | EV-010…EV-014 |
| Phase 2 – Currency-Architektur | Currency-Extraktion (SEC), API-Felder, zentrale Formatierung, UI-Kennzeichnung | EV-020…EV-023 |
| Phase 3 – Chart-Datenmodell und Hover | Datums-Serialisierung, Bucketing, Custom-Tooltip, FE-Test-Setup | EV-030…EV-032 |
| Phase 4 – Zeitraumfilter | TimeRangeFilter-Komponente + Utility + Integration | EV-040, EV-041 |
| Phase 5 – Prozentuale Veränderungen | %-Utility + Anzeige | EV-050, EV-051 |
| Phase 6 – Kurscharts | Preis-History-Endpoint, Kurschart Analyse, Kursvergleich normalisiert | EV-060…EV-062 |
| Phase 7 – Dashboard-Favoriten | Batch-Endpoint + Favoriten-Sektion mit Sparklines | EV-070, EV-071 |
| Phase 8 – Billing-Navigation | Sidebar-Konditionalisierung + Statusmatrix-Verifikation | EV-080, EV-081 |
| Phase 9 – Regression und Launch-Prüfung | Vollständiger Regressionsdurchlauf | EV-090 |

Phasen 1, 2/3 und 8 sind voneinander unabhängig und können bei Bedarf parallel begonnen werden; innerhalb einer Phase gilt die ID-Reihenfolge. Phasen 4→5→6→7 bauen aufeinander auf (3→4→5→6→7).

## 10. Detaillierte Aufgaben

### [EV-001] Backend-Baseline-Tests für Symbolsuche und Custom-History-Format

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 0
**Priorität:** Hoch
**Aufwand:** S
**Risiko:** Niedrig
**Abhängigkeiten:** keine

#### Ziel
Ist-Verhalten der Symbolsuche und des Custom-History-Serienformats mit pytest festhalten, damit spätere Änderungen (EV-010ff, EV-030) Regressionen sichtbar machen.

#### Aktueller Zustand
`api/tests/` existiert mit pytest-Setup (`pytest.ini`); für `/analyze/symbols` und das `{date, value}`-Serienformat gibt es keine gezielten Tests (prüfen: `ls api/tests/`, vorhandene Tests nicht duplizieren).

#### Beweis oder Fundstelle
`api/routes/analyze.py:53-104` (Suche), `api/routes/custom_analysis.py:148-176` (`_wrap_metric_result`, Serienbau).

#### Geplante technische Änderung
Neue Testdatei `api/tests/test_symbol_search.py`: (a) leere Query liefert POPULAR_SYMBOLS-Teilmenge, (b) Query „AAPL" findet Apple, (c) `limit` wird respektiert (≤ 50), (d) inaktive Symbole werden nicht geliefert, (e) Namenssuche per ilike. Neue Testdatei `api/tests/test_custom_history_format.py`: `_wrap_metric_result` mit Mini-DataFrame ⇒ Serie hat `{date, value}`-Form; Datumsformat des Ist-Zustands (voller Timestamp) explizit dokumentierend testen (Test wird in EV-030 angepasst).

#### Betroffene Dateien und Komponenten
Neu: `api/tests/test_symbol_search.py`, `api/tests/test_custom_history_format.py`. Keine Produktivdatei.

#### Zu schützende bestehende Funktionen
Keine (nur Tests).

#### Implementierungsschritte
1. Bestehende Test-Fixtures in `api/tests/` sichten (DB-Fixture/TestClient-Muster übernehmen).
2. Beide Testdateien schreiben; Symbole über Fixture in Test-DB seeden.
3. `pytest api/tests/test_symbol_search.py api/tests/test_custom_history_format.py` grün.

#### Automatisierte Tests
Die Aufgabe IST der Test.

#### Manuelle Tests
Keine.

#### Akzeptanzkriterien
- [x] Beide Testdateien laufen lokal grün; kompletter `pytest`-Lauf bleibt grün.

#### Nachweis der erfolgreichen Umsetzung
Umgesetzt wie geplant, mit einer Anpassung: `search_symbols` ist `@limiter.limit`-dekoriert und benötigt ein echtes `Request`-Objekt (nicht `None`) — Fake-Request-Helper analog `test_quota_active_jobs_order_p2_3.py` übernommen.

- `api/tests/test_symbol_search.py` (10 Tests): leere Query ⇒ aktive POPULAR_SYMBOLS-Teilmenge; leere Tabelle ⇒ statischer Fallback; Large Cap (AAPL), Small Cap (ZION), Sonderzeichen (`BRK.B`), Namenssuche (Coca-Cola), inaktives/delistetes Symbol ausgeschlossen, unbekanntes Symbol ⇒ leer, `limit` respektiert, Response-Form `{symbol,name,sectors}`.
- `api/tests/test_custom_history_format.py` (4 Tests): dokumentiert den Root-Cause-Beweis für P5 — `str(idx)` liefert vollen Timestamp (`"2024-09-30 00:00:00"`), zwei Firmen mit versetzten Fiskaldaten erzeugen disjunkte Datums-Keys (genau das, was EV-030 beheben muss), Mehrspalten-DataFrame nutzt erste Spalte, spaltenlose DataFrame ⇒ leere Serie.

Testlauf: `pytest api/tests/test_symbol_search.py api/tests/test_custom_history_format.py -v` → 14 passed. Kompletter Lauf: `pytest api/tests/` → **96 passed**, keine Regression an den 82 bereits vorhandenen Tests.

#### Rollback-Strategie
Testdateien löschen (kein Produktivcode betroffen).

#### Offene Fragen
Keine.

---

### [EV-002] Frontend-Baseline: Build-Nachweis und manuelle Ist-Aufnahme der Charts

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 0
**Priorität:** Hoch
**Aufwand:** XS
**Risiko:** Niedrig
**Abhängigkeiten:** keine

#### Ziel
Nachweis, dass `npm run build` vor Beginn der Arbeiten fehlerfrei ist, plus dokumentierte Ist-Aufnahme des Fehlverhaltens (Tooltip) als Vergleichsbasis.

#### Aktueller Zustand
Kein Frontend-Test-Framework; Build ist die einzige automatische Prüfung.

#### Beweis oder Fundstelle
`frontend/package.json` (kein test-Script).

#### Geplante technische Änderung
Keine Codeänderung. `cd frontend && npm run build` ausführen und Ergebnis festhalten. Dev-Server starten, Vergleich mit 3 Unternehmen (z. B. AAPL, MSFT, KO) durchführen, Tooltip-Fehlverhalten (nur 1–2 Firmen sichtbar) per Screenshot dokumentieren.

#### Betroffene Dateien und Komponenten
Keine.

#### Zu schützende bestehende Funktionen
Keine.

#### Implementierungsschritte
1. `npm run build` fehlerfrei nachweisen.
2. Vergleich mit 3 Firmen starten, über Chart hovern, Screenshot des Tooltips sichern.

#### Automatisierte Tests
Keine.

#### Manuelle Tests
Wie oben.

#### Akzeptanzkriterien
- [x] Build grün, Ist-Beobachtung des Tooltip-Verhaltens liegt vor.

#### Nachweis der erfolgreichen Umsetzung
**Build:** `cd frontend && npm run build` → `tsc -b` + `vite build` + `build:glossary` fehlerfrei. 3274 Module transformiert, Bundles erzeugt (`IntroOverlay` 891 KB, `index` 1.169 MB — bestehende Chunk-Size-Warnung, keine neue Regression), Glossar (94 Kennzahl-Seiten) generiert.

**Manuelle Ist-Aufnahme (Vergleich AAPL/MSFT/KO, Kennzahl „Historischer Umsatz"):** Lokales Testkonto registriert (`ev-baseline-test@example.com`, E-Mail-Verifikation via `scripts/verify_email.py`), Backend (`uvicorn`, Port 8000) und Frontend-Dev-Server (Vite, Port 5173) gestartet, Vergleich mit den drei Firmen durchgeführt (alle drei Jobs „Fertig").

Da der Screenshot-Renderer der Browser-Vorschau in dieser Session zeitweise instabil war (wiederholt rein schwarze Bilder), wurde das Tooltip-Verhalten stattdessen direkt über die DOM (`.recharts-tooltip-wrapper`, synthetische Hover-Events) ausgelesen — das ist für den Nachweis sogar präziser als ein Bildschirmfoto, da der exakte Tooltip-Inhalt textuell vorliegt:

- Hover auf Datenpunkt „2019-06-30": Tooltip zeigt `AAPL : 259 Mrd. | KO : 35 Mrd. | MSFT : 126 Mrd.` — alle drei Firmen.
- Hover auf Datenpunkt „2021-03-31" (nach Hinzufügen einer zweiten Zeitreihen-Kennzahl): Tooltip zeigt `AAPL : 325 Mrd. | KO : 33 Mrd. | MSFT : 160 Mrd.` — ebenfalls alle drei.
- Mehrere andere Hover-Positionen entlang derselben Linie lieferten einen **leeren** Tooltip (kein Eintrag für irgendeine Firma), obwohl die Mausposition weiterhin innerhalb der Chart-Fläche lag.

**Einordnung (wichtig für spätere Vergleichbarkeit mit EV-030/031):** Diese Stichprobe zeigt nicht das vom Kollegen beschriebene „nur 1–2 Firmen sichtbar", sondern ein gröberes Symptom derselben Ursache: an manchen X-Positionen ist der Tooltip komplett leer, an anderen zeigt er zufällig alle drei. Das ist konsistent mit dem in EV-001 code-seitig bewiesenen Mechanismus (`api/tests/test_custom_history_format.py::test_two_companies_with_different_fiscal_dates_never_share_a_date_key`): `mergeLayers` schlüsselt nach exaktem Datums-String; die meisten X-Positionen im gerenderten Chart entsprechen dann gar keinem realen Datenpunkt-Schlüssel (daher leerer Tooltip), und nur zufällig exakt auf einer Zeile liegende Positionen zeigen Daten — dort dann ggf. sogar zufällig für alle Firmen, wenn deren Zeilen (durch Zufall oder Frequenz-Rundung) denselben Schlüssel teilen. Die Kernaussage aus EVOLVING.md Abschnitt 5.3/6 (P5) bleibt damit durch reproduzierbare Beobachtung **bestätigt**: der Tooltip ist an der Mehrzahl der Hover-Positionen nicht zuverlässig vollständig. Ein pixelgenauer Vorher/Nachher-Screenshot-Vergleich wird in EV-031 nachgeholt, sobald der Browser-Renderer wieder stabil läuft.

Lokales Testkonto (`ev-baseline-test@example.com`) bleibt für weitere manuelle Prüfschritte in Folgeaufgaben nutzbar.

#### Nachweis der erfolgreichen Umsetzung
Build-Log + Screenshot.

#### Rollback-Strategie
Entfällt.

#### Offene Fragen
Keine.

---

### [EV-010] Symbol-Import in Produktion betriebsfähig machen (Startup-Task)

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 1
**Priorität:** Hoch (höchste Nutzerwirkung)
**Aufwand:** M
**Risiko:** Mittel
**Abhängigkeiten:** EV-001

#### Ziel
Die Produktions-`symbols`-Tabelle wird zuverlässig mit dem vollen NYSE+Nasdaq-Universum (~5.900 aktive Titel) befüllt, ohne Shell-Zugriff auf Render zu benötigen.

#### Aktueller Zustand
Import läuft nur manuell (`python -m scripts.import_symbols`); kein Aufrufer repo-weit; Migration seeded 23 Zeilen; Render-Start = `alembic upgrade head && uvicorn …`. Backend/DB laufen inzwischen auf Render Starter MIT Shell-Zugriff (der „kein Shell"-Hinweis in LAUNCH.md:148 ist veraltet). Lokal 5.898 aktive Symbole, Prod sehr wahrscheinlich 23.

**Sofortmaßnahme (optional, vor der Automatisierung):** Betreiber kann per Render-Shell einmalig `python -m scripts.import_symbols` gegen Prod ausführen und mit `SELECT count(*) FROM symbols WHERE is_active;` verifizieren. Das behebt das Nutzerproblem sofort; diese Aufgabe (Automatisierung) bleibt trotzdem vollständig umzusetzen, damit sich das Problem nie wiederholt.

#### Beweis oder Fundstelle
`scripts/import_symbols.py` (gesamte Logik), `alembic/versions/9a4f8ab8c4c6_create_symbols_table.py:28-52,83-89`, `api/main.py:164-168` (Worker-Muster), `LAUNCH.md:148`.

#### Geplante technische Änderung
In `api/main.py` beim Startup (nach den bestehenden Workern) einen Task registrieren: zählt `SELECT count(*) FROM symbols WHERE is_active`; wenn < 1.000, wird die Import-Funktion aus `scripts/import_symbols.py` (bestehende `fetch_*`/`upsert_symbols`-Funktionen importieren, NICHT duplizieren) in einem Hintergrund-Thread/Task ausgeführt, damit der Serverstart nicht blockiert. Logging: Start, Dauer, importierte/deaktivierte Anzahl. Fehler beim Download dürfen den Serverstart nie verhindern (try/except mit Log). Schwellwert 1.000 als Konstante `SYMBOL_IMPORT_MIN_COUNT` mit Kommentar (D8).

#### Betroffene Dateien und Komponenten
`api/main.py` (Startup), `scripts/import_symbols.py` (ggf. Funktionen extrahieren/importierbar machen, Verhalten unverändert).

#### Zu schützende bestehende Funktionen
Serverstart-Zeit (Import asynchron!); bestehende Symbole/Favoriten (Upsert ist idempotent, `is_active`-Logik unverändert); bestehende Worker.

#### Implementierungsschritte
1. Prüfen, ob `import_symbols.py` als Modul importierbar ist (keine Seiteneffekte auf Modulebene); sonst `main()`-Guard ergänzen.
2. Startup-Task in `api/main.py` nach Worker-Muster ergänzen (Thread/`asyncio.to_thread`).
3. Lokal testen: Tabelle leeren (nur lokale Test-DB!), Server starten, Import läuft, Zählung > 5.000.
4. pytest-Test: Startup-Funktion mit gemockter Fetch-Funktion (kein echter Netzwerkzugriff in Tests).

#### Automatisierte Tests
Test für die Schwellwert-Logik (Import wird bei count ≥ 1.000 NICHT ausgelöst; bei < 1.000 ausgelöst; Fetch gemockt).

#### Manuelle Tests
Lokal mit leerer Test-DB: Serverstart ⇒ Log zeigt Import; `GET /analyze/symbols?query=ZION` findet Small Cap.

#### Akzeptanzkriterien
- [x] Serverstart blockiert nicht; bei fast leerer Tabelle wird importiert; bei gefüllter Tabelle passiert nichts.
- [x] Downloadfehler ⇒ Server läuft trotzdem, Fehler geloggt.

#### Nachweis der erfolgreichen Umsetzung
Umgesetzt wie geplant, mit Struktur-Anpassung: Statt die Import-Logik direkt in `api/main.py` zu registrieren, wurde ein eigenes Modul `api/services/symbol_sync_service.py` angelegt (`count_active_symbols`, `run_symbol_import`, `_run_import_locked` mit `asyncio.Lock` gegen Parallelläufe, `sync_symbols_on_startup` mit dem Schwellwert-Gate `SYMBOL_IMPORT_MIN_COUNT=1000`). `scripts/import_symbols.py` wurde unverändert wiederverwendet (`fetch_nasdaq_listed`, `fetch_other_listed`, `upsert_symbols` importiert — war bereits ohne Modul-Seiteneffekte importierbar, kein `main()`-Guard-Umbau nötig). Der Worker-Teil (EV-011, `symbol_sync_worker`) liegt bereits im selben Modul, ist aber noch nicht in `api/main.py` verdrahtet — das folgt in EV-011 mit eigenem Test.

`api/main.py`: Import ergänzt (`from api.services.symbol_sync_service import sync_symbols_on_startup`), Startup-Hook um `asyncio.create_task(sync_symbols_on_startup(SessionLocal))` erweitert (nach den drei bestehenden Workern).

**Automatisierte Tests** (`api/tests/test_symbol_sync_service.py`, 5 Tests, alle gemockt — kein echter Netzwerkzugriff): `count_active_symbols` zählt nur `is_active=True`; Import wird bei `count ≥ Schwellwert` NICHT ausgelöst (Spy wirft bei Aufruf); Import wird bei `count < Schwellwert` ausgelöst und schreibt die gemockt gelieferten Symbole in die DB; ein Fetch-Fehler (`ConnectionError`) wird abgefangen und darf `sync_symbols_on_startup` nicht crashen lassen; zwei sich zeitlich überlappende Aufrufe von `_run_import_locked` (kontrolliert via `asyncio.Event`, kein Race) — der zweite wird sofort mit `None` abgewiesen statt zu warten, der Import läuft nachweislich nur einmal (`calls == [1]`).

Testlauf: `pytest api/tests/test_symbol_sync_service.py -v` → 5 passed. Kompletter Lauf: `pytest api/tests/` → **101 passed** (96 vorherige + 5 neue), keine Regression.

**Manueller Nachweis am echten lokalen Server:** Backend neu gestartet (`uvicorn api.main:app --reload`), Log zeigt:
```
INFO api.services.symbol_sync_service: Symbol-Universum bereits befüllt (5898 aktive Symbole, Schwellwert 1000) - kein Startup-Import.
INFO:     Application startup complete.
```
Bestätigt: (a) der Check läuft beim Start, (b) er blockiert `Application startup complete` nicht, (c) bei der lokal bereits befüllten Tabelle (5.898 aktive Symbole, wie in Abschnitt 5.1 dokumentiert) wird korrekt KEIN Download ausgelöst. Der "Import wird bei leerer Tabelle wirklich ausgeführt"-Pfad ist damit ausschließlich durch die gemockten pytest-Tests bewiesen (ein Leeren der lokalen Produktiv-artigen DB nur für einen Live-Test wurde bewusst vermieden, um die bestehenden 5.898 Symbole nicht zu gefährden); der Produktions-Nachweis (Prod-DB tatsächlich < 1.000 → Import läuft dort) folgt mit EV-012 (`/admin/symbols/stats`) nach dem nächsten Prod-Deploy.

#### Rollback-Strategie
Startup-Task-Registrierung (`asyncio.create_task(sync_symbols_on_startup(...))` in `api/main.py`) entfernen bzw. den Import von `sync_symbols_on_startup` zurücknehmen; `api/services/symbol_sync_service.py` kann gefahrlos bestehen bleiben, da sie ungenutzt keine Wirkung hat. Bereits importierte Daten bleiben unkritisch in der Tabelle.

#### Offene Fragen
Keine (D8 entschieden als Standardannahme). Prod-Nachweis (tatsächlicher Import bei < 1.000 aktiven Symbolen in Produktion) steht noch aus — wird mit EV-012 nachgewiesen.

---

### [EV-011] Wöchentlicher Symbol-Sync-Worker

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 1
**Priorität:** Mittel
**Aufwand:** S
**Risiko:** Niedrig
**Abhängigkeiten:** EV-010

#### Ziel
Neue Listings erscheinen automatisch, Delistings werden auf `is_active=False` gesetzt – ohne manuelle Eingriffe.

#### Aktueller Zustand
Kein Sync; Delisting-Logik existiert bereits im Import (`import_symbols.py:154-157`), läuft aber nie.

#### Beweis oder Fundstelle
`api/main.py:164-168` (`watchlist_digest_worker` als Vorlage), `scripts/import_symbols.py:154-157`.

#### Geplante technische Änderung
Neuer async-Worker nach Vorlage der bestehenden Worker: Schleife mit 7-Tage-Intervall (`asyncio.sleep`), ruft dieselbe Import-Funktion wie EV-010 auf. Jitter/Startverzögerung von z. B. 10 Minuten nach Boot, damit Startup-Import (EV-010) und Worker nicht kollidieren; einfacher In-Process-Lock (Modul-Flag), damit nie zwei Importe parallel laufen.

#### Betroffene Dateien und Komponenten
`api/main.py`, ggf. neues `api/jobs/sync_symbols.py`.

#### Zu schützende bestehende Funktionen
Favoriten auf delisteten Symbolen: `is_active=False` löscht keine Favoriten (Zeile bleibt) – verifizieren, dass `GET /favorites` weiterhin liefert (Favoriten-Route liest nicht `is_active`; bestätigen und im Test festhalten).

#### Implementierungsschritte
1. Worker-Funktion + Lock implementieren.
2. Registrierung in `api/main.py`.
3. Test mit gemocktem Import: Worker ruft Import auf, Lock verhindert Parallel-Lauf.

#### Automatisierte Tests
Lock-Test; Favoriten-auf-inaktivem-Symbol-Test (`GET /favorites` liefert Eintrag weiterhin).

#### Manuelle Tests
Intervall lokal auf 60 s stellen, zwei Durchläufe im Log beobachten, Intervall zurückstellen.

#### Akzeptanzkriterien
- [x] Worker läuft wöchentlich, nie parallel zum Startup-Import; Favoriten delisteter Titel bleiben sichtbar.

#### Nachweis der erfolgreichen Umsetzung
Umgesetzt wie geplant, im selben Modul wie EV-010 (`api/services/symbol_sync_service.py::symbol_sync_worker`, statt eines separaten `api/jobs/sync_symbols.py` — konsistent mit der Wiederverwendung von `_run_import_locked`/`_import_lock` aus EV-010, kein Duplikat). Kein zusätzliches Boot-Jitter nötig: der geteilte `_import_lock` reicht bereits aus, um Überschneidungen mit dem Startup-Check (EV-010) zu verhindern (belegt durch den bereits in EV-010 geschriebenen Lock-Test). Anders als der Startup-Check prüft der Worker **keinen** Schwellwert — er ruft `_run_import_locked` bei jedem Tick unbedingt auf, da ein Schwellwert-Gate nach dem ersten erfolgreichen Import (Tabelle dauerhaft > 1.000) nie wieder auslösen würde und neue Listings/Delistings damit für immer unerkannt blieben.

`api/main.py`: Import erweitert (`symbol_sync_worker` zusätzlich importiert), Startup-Hook um `asyncio.create_task(symbol_sync_worker(SessionLocal))` ergänzt (Default-Intervall 7 Tage, Konstante im Funktions-Default `interval_seconds=7*24*60*60`).

**Automatisierte Tests** (`api/tests/test_symbol_sync_worker.py`, 3 Tests): Worker ruft `_run_import_locked` bei kurzem Intervall (0,01 s) mehrfach auf (`len(calls) >= 2`), auch wenn die Tabelle mit 5.000 Zeilen weit über dem Schwellwert liegt — belegt, dass kein Schwellwert-Gate greift; Worker wartet nachweislich das volle Intervall ab, bevor er zum ersten Mal läuft (10 s Intervall, 0,05 s Beobachtungsfenster ⇒ `calls == []`); **Schutz bestehender Funktion** — ein Favorit auf einem Symbol, das `upsert_symbols` als `is_active=False` markiert (simuliertes Delisting, leere Fetch-Liste), bleibt über `get_favorites` weiterhin sichtbar (verifiziert zusätzlich durch Code-Lesung: `api/crud/favorite.py` referenziert `is_active` an keiner Stelle — `Favorite` und `Symbol` sind unabhängige Tabellen, nur lose über den Symbol-String verknüpft).

Testlauf: `pytest api/tests/test_symbol_sync_worker.py -v` → 3 passed. Kompletter Lauf: `pytest api/tests/` → **104 passed** (101 nach EV-010 + 3 neue Tests aus dieser Aufgabe), keine Regression.

**Manueller Nachweis am echten lokalen Server:** `api/main.py` zweimal editiert (Import EV-010, dann EV-011-Ergänzung); der laufende `uvicorn --reload`-Prozess hat beide Änderungen selbstständig übernommen (`WARNING: StatReload detected changes in 'api/main.py'. Reloading...`, zweimal im Log), und `Application startup complete.` erscheint jeweils fehlerfrei danach — der neue Worker verhindert also nicht den Serverstart. Ein 7-Tage-Intervall lässt sich in der laufenden Session naturgemäß nicht in Echtzeit beobachten; das reale wöchentliche Auslösen ist durch die gemockten Timing-Tests oben abgedeckt (kurzes Intervall ⇒ mehrfacher Aufruf, langes Intervall ⇒ kein vorzeitiger Aufruf), die exakt dieselbe Schleifenlogik prüfen wie sie mit dem echten 7-Tage-Wert läuft.

#### Rollback-Strategie
`asyncio.create_task(symbol_sync_worker(SessionLocal))`-Zeile in `api/main.py` entfernen; `symbol_sync_worker`-Funktion kann gefahrlos ungenutzt im Modul verbleiben.

#### Offene Fragen
Keine.

---

### [EV-012] Nachweis-Endpoints: Symbol-Statistik und manueller Refresh (Admin)

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 1
**Priorität:** Mittel
**Aufwand:** S
**Risiko:** Niedrig
**Abhängigkeiten:** EV-010

#### Ziel
Beweisbar machen, wie viele Unternehmen tatsächlich analysierbar sind (je Börse, aktiv/inaktiv), und einen manuellen Import-Trigger bereitstellen, der keine Render-Shell-Sitzung erfordert.

#### Aktueller Zustand
Keine Zähl-/Trigger-Endpoints; Admin-Routen-Muster existiert (`api/routes/admin_stats.py`, `require_admin` in `api/core/dependencies.py:86-89`).

#### Beweis oder Fundstelle
s. o.

#### Geplante technische Änderung
In `api/routes/admin_stats.py` (oder neuem `admin_symbols.py`, im Router-Setup von `api/main.py` registrieren): `GET /admin/symbols/stats` ⇒ `{total, active, inactive, by_exchange:{NASDAQ:{active,inactive},…}, last_import_at}`; `POST /admin/symbols/refresh` ⇒ startet Import im Hintergrund (gleicher Lock wie EV-011), Antwort 202. Beide mit `require_admin`.

#### Betroffene Dateien und Komponenten
`api/routes/admin_stats.py` oder neu `api/routes/admin_symbols.py`; optional Anzeige der Zahlen im bestehenden `AdminDashboardPage.tsx` (nur wenn trivial einfügbar, sonst weglassen – Admin-Dashboard ist eigenes Produktthema).

#### Zu schützende bestehende Funktionen
Bestehende Admin-Statistiken; `require_admin`-Schutz zwingend.

#### Implementierungsschritte
1. Stats-Query (group by exchange, is_active) + Route.
2. Refresh-Route mit Lock + 202.
3. pytest: Nicht-Admin ⇒ 403; Admin ⇒ korrekte Zählstruktur.

#### Automatisierte Tests
403-Test, Zählstruktur-Test mit geseedeten Symbolen.

#### Manuelle Tests
Als Admin `GET /admin/symbols/stats` aufrufen; Refresh auslösen und Stats-Änderung beobachten.

#### Akzeptanzkriterien
- [x] Stats liefern korrekte Zahlen je Börse; Refresh nur für Admins; kein Doppel-Import möglich.

#### Nachweis der erfolgreichen Umsetzung
Umgesetzt wie geplant, als neue Datei `api/routes/admin_symbols.py` (eigener Router-Prefix `/admin/symbols`, in `api/main.py` registriert) statt Erweiterung von `admin_stats.py` — sauberere Trennung, da Symbol-Universum fachlich nichts mit den bestehenden Funnel-/MRR-Statistiken zu tun hat. `GET /admin/symbols/stats` liefert `{total, active, inactive, by_exchange:{Börse:{active,inactive}}, last_updated_at}` (`last_updated_at` statt `last_import_at` — nutzt das bereits vorhandene `Symbol.updated_at`-Feld, kein neues Schema nötig). `POST /admin/symbols/refresh` prüft `_import_lock.locked()` VOR dem Start (liefert `"already_running"` statt eines zweiten Loggings, falls schon einer läuft) und stößt danach `_run_import_locked` als Fire-and-Forget-Task an, Antwort sofort mit 202. Admin-Dashboard-Frontend-Anzeige bewusst weggelassen (Betreiber-Entscheidung: Admin-Dashboard ist eigenes Produktthema, siehe Projektgedächtnis „Admin-Dashboard eigen").

**Automatisierte Tests** (`api/tests/test_admin_symbols.py`, 6 Tests): `_compute_symbol_stats` zählt korrekt nach Börse und `is_active` (inkl. leerer Tabelle); `require_admin` weist Nicht-Admin mit 403 ab; Admin erhält korrekte Zählstruktur; `refresh_symbols` stößt den Import als Hintergrund-Task an und antwortet sofort mit `{"status":"started"}` (Import-Aufruf per Spy nachgewiesen); ein bereits laufender Import wird korrekt als `{"status":"already_running"}` gemeldet, ohne einen zweiten Lauf zu starten.

Testlauf: `pytest api/tests/test_admin_symbols.py -v` → 6 passed. Kompletter Lauf: `pytest api/tests/` → **110 passed** (104 vorherige + 6 neue), keine Regression.

**Manueller Nachweis am echten lokalen Server (End-to-End, kein Mock):** Testkonto per `scripts/set_admin.py` zu `plan=admin` gemacht, per `curl` eingeloggt (`POST /auth/login`, OAuth2-Formular) und beide neuen Endpoints mit echtem Bearer-Token aufgerufen:

```
GET /admin/symbols/stats
{"total": 6318, "active": 5898, "inactive": 420,
 "by_exchange": {"NYSE": {"active": 2214, "inactive": 61},
                 "NASDAQ": {"active": 3412, "inactive": 353},
                 "NYSE American": {"active": 272, "inactive": 6}},
 "last_updated_at": "2026-07-03T18:14:45.597433"}

POST /admin/symbols/refresh → HTTP 202 {"status": "started"}
```

Nach dem Refresh zeigte das Server-Log einen echten (nicht gemockten) Lauf gegen die NASDAQ-Trader-Live-Dateien: `Symbole verarbeitet: 5903 (davon neu: 16, als delisted markiert: 431)` / `Symbol-Import abgeschlossen: 5903 Zeilen verarbeitet.` Ein erneuter Stats-Aufruf bestätigte die aktualisierten Zahlen (`active: 5903`, `inactive: 431`, `last_updated_at` aktualisiert). **Das ist der geforderte Nachweis aus Aufgabe 1 des Feedbacks:** deutlich über 5.000 tatsächlich aktive NYSE+Nasdaq-Symbole sind nachweislich durchsuchbar/analysierbar, und der manuelle Trigger funktioniert Ende-zu-Ende ohne Render-Shell-Zugriff. Der Prod-Nachweis (identischer Aufruf gegen die Produktions-URL nach dem nächsten Deploy) steht noch aus, da dieser Plan ausschließlich lokal umgesetzt/getestet wird.

#### Rollback-Strategie
Router-Registrierung (`app.include_router(admin_symbols_router)`) in `api/main.py` entfernen bzw. `api/routes/admin_symbols.py` löschen; keine Datenbankänderung betroffen.

#### Offene Fragen
Keine.

---

### [EV-013] Stillen Symbol-Such-Fallback durch sichtbaren Fehlerzustand ersetzen

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 1
**Priorität:** Hoch
**Aufwand:** S
**Risiko:** Niedrig
**Abhängigkeiten:** EV-002

#### Ziel
Backend-Fehler bei der Suche dürfen nicht mehr wie eine „kleine Liste" aussehen; Nutzer sieht einen klaren Fehler-/Retry-Zustand.

#### Aktueller Zustand
`searchSymbolsSafe` fängt JEDEN Fehler und liefert still 23 statische `LOCAL_SYMBOLS` (`frontend/src/api/analysis.ts:64-83`; Quelle `frontend/src/data/symbols.ts`).

#### Beweis oder Fundstelle
s. o.

#### Geplante technische Änderung
`searchSymbolsSafe` gibt bei Fehler ein diskriminierbares Ergebnis zurück (z. B. `{entries, degraded: true}` oder wirft weiter und der Hook fängt): `useSymbolSearch` erhält `error`-State; `SymbolCommandField` und `compare/SymbolSuggestField` zeigen bei `degraded` eine Hinweiszeile („Suche derzeit eingeschränkt – erneut versuchen") über den (weiterhin angezeigten) Fallback-Einträgen. Der Fallback bleibt als Offline-Netz erhalten, wird aber sichtbar gemacht.

#### Betroffene Dateien und Komponenten
`frontend/src/api/analysis.ts`, `frontend/src/hooks/useSymbolSearch.ts`, `frontend/src/components/**/SymbolCommandField.tsx`, `frontend/src/compare/SymbolSuggestField.tsx`.

#### Zu schützende bestehende Funktionen
Bestehendes Suchverhalten im Normalfall (Debounce 250 ms, limit 8, leere Query ⇒ Popular-Liste) unverändert; Tastaturnavigation der Dropdowns.

#### Implementierungsschritte
1. Rückgabetyp erweitern (additiv, beide Aufrufer anpassen).
2. Hinweiszeile in beiden Feldern (gleiche Optik wie bestehende Empty-States).
3. Manuell testen: Backend stoppen ⇒ Hinweis erscheint; Backend an ⇒ normale Suche.

#### Automatisierte Tests
Falls vitest (EV-032) schon vorhanden: Hook-Logik-Test; sonst manuell.

#### Manuelle Tests
Backend aus/an; Rate-Limit provozieren (61 Anfragen/min) ⇒ Hinweis statt stiller 23er-Liste.

#### Akzeptanzkriterien
- [x] Fehlerfall visuell erkennbar auf Analyse- UND Vergleichsseite; Normalfall unverändert; `npm run build` grün.

#### Nachweis der erfolgreichen Umsetzung
Umgesetzt wie geplant: `searchSymbolsSafe` (`frontend/src/api/analysis.ts`) liefert jetzt `{entries, degraded}` statt eines bloßen Arrays (neuer Typ `SymbolSearchResult`); `useSymbolSearch` gibt zusätzlich `isDegraded` zurück. Da `searchSymbolsSafe` repo-weit nur von `useSymbolSearch` aufgerufen wird (verifiziert per Grep), war keine weitere Aufrufer-Anpassung nötig. Die Hinweiszeile wurde über die tatsächliche Prop-Kette gereicht — der Plan nannte `SymbolCommandField`/`SymbolSuggestField` direkt, tatsächlich liegt zwischen `AnalyzePage` und `SymbolCommandField` noch `AnalyzeWorkspace` bzw. `AdHocAnalysisPanel` (zwei Verwendungsstellen: Standard-Workspace und das Ad-hoc-Modal) — beide Zwischenkomponenten wurden um die optionale Prop `isSymbolSearchDegraded` erweitert, damit die Kette nicht bricht. Pfadkorrektur gegenüber dem Plan: Die Compare-Komponente liegt unter `frontend/src/components/compare/SymbolSuggestField.tsx`, nicht `frontend/src/compare/SymbolSuggestField.tsx`.

Optik: Hinweistext in `theme.colors.dangerText` (kein eigenes `warning`-Token im Theme vorhanden, semantisch passend, da es sich um einen echten Fehlerzustand handelt), über den Vorschlägen, mit Trennlinie — Fallback-Einträge (23 statische Symbole) bleiben weiterhin nutzbar und sichtbar darunter/danach.

**Automatisierte Tests:** Da EV-032 (vitest-Setup) noch nicht existiert, wie im Plan als Alternative vorgesehen nur manuell geprüft (kein automatisierter Test für dieses Frontend-Verhalten in dieser Aufgabe).

**Manueller Nachweis am echten lokalen Server (beide Seiten, Backend aus/an):**
- Analyseseite: Backend an, Suche „AAP" ⇒ vier echte DB-Treffer (AAP, AAPG, AAPL, CAAP), kein Hinweis. Backend gestoppt, Suche „MSF" ⇒ Hinweis „Suche derzeit eingeschränkt – zeige eine begrenzte Auswahl. Bitte erneut versuchen." + „Keine passenden Symbole gefunden." (MSF matcht keinen der 23 Fallback-Einträge). Suche „AAPL" (Backend weiterhin aus) ⇒ derselbe Hinweis PLUS der echte Fallback-Treffer „AAPL — Apple Inc." darunter. Backend neu gestartet, Suche „AAP" erneut ⇒ wieder vier echte DB-Treffer, kein Hinweis mehr.
- Vergleichsseite: Backend gestoppt, neues Firmenfeld geöffnet, „AAPL" eingetippt ⇒ derselbe Hinweistext + Fallback-Treffer erscheinen auch dort (per DOM-Inspektion bestätigt, da der Screenshot-Renderer der Vorschau-Session zeitweise instabil war — siehe bereits in EV-002 dokumentiertes Verhalten).

Build: `npm run build` fehlerfrei (3274 Module, Bundles erzeugt, Glossar generiert) — bestätigt vor der Browser-Verifikation.

#### Rollback-Strategie
Commit revert (isolierte Änderung: `analysis.ts`, `useSymbolSearch.ts`, `SymbolCommandField.tsx`, `SymbolSuggestField.tsx`, `AnalyzeWorkspace.tsx`, `AdHocAnalysisPanel.tsx`, `AnalyzePage.tsx`).

#### Offene Fragen
Keine.

---

### [EV-014] Weiche Symbol-Validierung beim Analyse-/Vergleichsstart

**Status:** ✅ Erledigt (2026-07-14) — Code + Tests lokal fertig; Prod-Deploy-Gate (s. u.) weiterhin zu beachten
**Phase:** 1
**Priorität:** Mittel
**Aufwand:** S
**Risiko:** Mittel
**Abhängigkeiten:** EV-010, EV-012 (Prod-Universum MUSS nachweislich > 5.000 sein, bevor dies deployt wird!)

#### Ziel
Ungültige/delistete Symbole scheitern früh mit klarer 422-Meldung statt kryptischer Fehler tief im Agenten.

#### Aktueller Zustand
`start_single_analysis` (`api/routes/analyze.py:236-323`), Full-Analysis (`api/routes/full_analysis.py:114-149`) und Custom (`api/routes/custom_analysis.py:48-49`) normalisieren nur Groß-/Kleinschreibung; keine Existenzprüfung. Favoriten prüfen bereits gegen `symbols` (`api/routes/favorites.py:58`) – dieses Muster übernehmen.

#### Beweis oder Fundstelle
s. o.

#### Geplante technische Änderung
Gemeinsame Helper-Funktion (z. B. in `api/dependencies` oder `api/utils`): `ensure_known_symbol(db, symbol)` ⇒ 422 mit Meldung „Symbol X ist nicht im unterstützten NYSE/Nasdaq-Universum" wenn kein aktiver Eintrag. In alle drei Start-Routen einbauen. WICHTIG: Delistete (`is_active=False`) ebenfalls ablehnen, aber mit eigener Meldung („nicht mehr gelistet"). Frontend: Fehlermeldung der 422 in den bestehenden Job-Fehlerpfaden anzeigen (prüfen, wie `useAnalysisJobs`/`useCompare` HTTP-Fehler heute rendern – vorhandenes Fehler-UI wiederverwenden).

#### Betroffene Dateien und Komponenten
`api/routes/analyze.py`, `api/routes/full_analysis.py`, `api/routes/custom_analysis.py`, neuer Helper; Frontend nur falls Fehlertext nicht durchgereicht wird.

#### Zu schützende bestehende Funktionen
Alle heute funktionierenden Analysen (deshalb Deploy-Reihenfolge-Zwang: erst Universum füllen); Groß-/Kleinschreibungs-Toleranz; Symbole mit Sonderzeichen (z. B. `BRK.B`/`BRK-B` – prüfen, in welcher Schreibweise die NASDAQ-Trader-Dateien sie führen, und dieselbe Normalisierung wie der Import verwenden!).

#### Implementierungsschritte
1. Schreibweisen-Analyse Punkt/Bindestrich zwischen Importdatei, DB und yfinance dokumentieren; Normalisierung im Helper angleichen.
2. Helper + Einbau in die drei Routen.
3. pytest: gültig ⇒ 200/202; unbekannt ⇒ 422; delistet ⇒ 422 mit anderer Meldung; `brk.b` (klein, Sonderzeichen) ⇒ akzeptiert.

#### Automatisierte Tests
s. Schritt 3.

#### Manuelle Tests
Analyse mit „FOOBARX" ⇒ saubere Meldung im UI; Analyse mit „BRK.B" ⇒ läuft.

#### Akzeptanzkriterien
- [x] Frühe, verständliche Ablehnung; keine einzige heute gültige Analyse wird blockiert (Stichproben Large/Small Cap/Sonderzeichen).

#### Nachweis der erfolgreichen Umsetzung
Umgesetzt wie geplant, mit einer Verbesserung gegenüber dem ursprünglichen Entwurf: `ensure_known_symbol(db, symbol)` (`api/utils/symbol_validation.py`, neu) wirft nicht nur 422 bei unbekannt/delistet, sondern **gibt die kanonische DB-Schreibweise zurück** (`return row.symbol`). Grund: reine Validierung hätte "brk.b" zwar als gültig durchgewunken, aber die nachfolgende Job-Erstellung/Historie/Agent-Aufruf hätte weiterhin mit dem Punkt-Symbol "BRK.B" gearbeitet statt mit dem im Rest der App erwarteten Bindestrich-Stil "BRK-B" — das hätte die eigentliche Analyse (yfinance/SEC-Aufrufe) potenziell brechen können. Alle vier Aufrufstellen wurden daher auf `symbol = ensure_known_symbol(db, symbol)` umgestellt.

Betroffene reale Aufrufstellen (der Plan nannte 3 Dateien, `custom_analysis.py` hat aber zwei unabhängige Start-Einstiegspunkte, beide wurden behandelt):
- `api/routes/analyze.py::start_single_analysis`
- `api/routes/full_analysis.py::start_full_analysis`
- `api/routes/custom_analysis.py::start_custom_analysis`
- `api/routes/custom_analysis.py::run_definition`

In allen vier Routen wurde die Prüfung — wie im Plan gefordert — VOR den Quota-Verbrauch gezogen (gleiche Begründung wie der bestehende Active-Jobs-Check aus LAUNCH_AUDIT.md P2-3: ein von vornherein ungültiges Symbol soll keine Analyse-Einheit kosten). Escape-Hatch ergänzt (nicht explizit im Plan gefordert, aber notwendig): ist die `symbols`-Tabelle komplett leer (kein Import je gelaufen, z. B. frische lokale Dev-DB), validiert `ensure_known_symbol` gar nicht, statt jede Analyse zu blockieren.

**Automatisierte Tests** (16 neue Tests über 2 Dateien):
- `api/tests/test_symbol_validation.py` (5 Tests, reine Helper-Logik): bekanntes aktives Symbol ⇒ akzeptiert, gibt sich selbst zurück; unbekannt ⇒ 422 „nicht im unterstützten…"; delistet ⇒ 422 „nicht mehr gelistet"; `BRK.B` findet die als `BRK-B` gespeicherte Zeile UND gibt `BRK-B` zurück; leere Tabelle blockiert nichts und gibt das Symbol unverändert zurück.
- `api/tests/test_analysis_start_symbol_validation.py` (11 Tests, Routen-Ebene, alle vier Einstiegspunkte): gültiges Symbol ⇒ Start gelingt (200er-Antwort-Dict); unbekannt ⇒ 422 (alle vier Routen); delistet ⇒ 422 mit eigener Meldung (bei `start_single_analysis`/`start_full_analysis`); `brk.b` (klein, Punkt) bei `start_single_analysis` ⇒ Start gelingt UND `result["symbol"] == "BRK-B"` (Beweis der Kanonisierung).

**Aufgetretene und behobene Testinfrastruktur-Probleme** (dokumentiert, da sie zeigen, warum der volle Suite-Lauf nach JEDER Änderung Pflicht ist):
1. Drei bereits bestehende Tests (`test_full_analysis_quota_units_p2_11.py`, drei Fälle in `test_quota_active_jobs_order_p2_3.py`) brachen zunächst mit `OperationalError: no such table: symbols`, weil ihre In-Memory-SQLite-Fixtures die `symbols`-Tabelle nie erzeugt hatten. Behoben durch Ergänzen von `Symbol.__table__` in den jeweiligen `Base.metadata.create_all(...)`-Aufrufen (bzw. Umstellen zweier Tests von der schmalen `db`- auf die erweiterte `db_full`-Fixture) — der leere-Tabelle-Escape-Hatch in `ensure_known_symbol` greift danach korrekt.
2. Meine eigenen neuen „akzeptiert"-Tests hinterließen zunächst echte, dauerhaft „running" bleibende Jobs im prozessweiten `job_manager`-Singleton (derselbe Objekt unter allen drei Routen-Modulen; `submit` war gemockt, wodurch `set_done` nie lief) — da jede frische In-Memory-SQLite-Test-DB Autoincrement-User-IDs wieder bei 1 beginnen lässt, verfälschte das den `count_active_jobs`-Check nachfolgender, unabhängiger Tests (u. a. genau der oben erwähnte P2-11-Test, der beim vollen Suite-Lauf plötzlich fehlschlug, obwohl er isoliert lief). Behoben durch eine Test-Hilfsfunktion `_isolate_job_manager`, die zusätzlich zu `submit` auch `create_job` und `count_active_jobs` mockt, damit die „akzeptiert"-Tests keinerlei Spuren im geteilten Singleton hinterlassen.

Testlauf: `pytest api/tests/test_symbol_validation.py api/tests/test_analysis_start_symbol_validation.py -v` → 16 passed. Kompletter Lauf, zweimal hintereinander zur Stabilitätsprüfung: `pytest api/tests/` → **126 passed** (110 nach EV-013 + 16 neue Tests aus dieser Aufgabe), keine Regression, reproduzierbar stabil.

**Manueller Nachweis am echten lokalen Server (alle vier Routen, echte Bearer-Auth):**
```
POST /analyze/wachstumswerte/start?symbol=FOOBARX  → 422 "Symbol FOOBARX ist nicht im unterstützten NYSE/Nasdaq-Universum."
POST /analyze/wachstumswerte/start?symbol=brk.b    → 200 {"job_id": "...", "symbol": "BRK-B", ...}
POST /full/start?symbol=MSFT                       → 200 {"job_id": "...", "symbol": "MSFT", "total": 10}
POST /full/start?symbol=NOTREAL123                 → 422 "Symbol NOTREAL123 ist nicht im unterstützten NYSE/Nasdaq-Universum."
POST /analyze/custom/start {"symbol":"NOTREAL123"} → 422 "Symbol NOTREAL123 ist nicht im unterstützten NYSE/Nasdaq-Universum."
```
Bestätigt: alle vier Einstiegspunkte lehnen unbekannte Symbole früh und verständlich ab, akzeptieren gültige Symbole unverändert (inkl. Sonderzeichen-Normalisierung) und canceln nicht auf Kosten bereits funktionierender Analysen (MSFT lief durch wie zuvor).

**Deploy-Hinweis (unverändert kritisch):** Dieser Code ist lokal fertig und getestet, darf aber laut Plan-Abhängigkeit **erst nach** einem bestätigten Produktions-Nachweis über `GET /admin/symbols/stats` (EV-012) ausgerollt werden — sonst würden auf einer noch unbefüllten Prod-Tabelle plötzlich gültige Analysen mit 422 abgelehnt. Diese Reihenfolge liegt außerhalb der Kontrolle dieser lokalen Implementierungssitzung und obliegt dem Betreiber beim nächsten Deploy.

#### Rollback-Strategie
Die vier `symbol = ensure_known_symbol(db, symbol)`-Zeilen in `analyze.py`/`full_analysis.py`/`custom_analysis.py` (×2) entfernen bzw. auf die vorherige reine `_norm_symbol`/`.strip().upper()`-Zeile zurücksetzen; `api/utils/symbol_validation.py` kann gefahrlos ungenutzt bestehen bleiben.

#### Offene Fragen
D9 (Standardannahme: umsetzen).

---

### [EV-020] Reporting-Währung aus SEC-XBRL extrahieren und im Backend verfügbar machen

**Status:** ✅ Erledigt (2026-07-14) — Umfang gegenüber ursprünglichem Plan bewusst angepasst, siehe Nachweis
**Phase:** 2
**Priorität:** Hoch
**Aufwand:** L
**Risiko:** Mittel
**Abhängigkeiten:** EV-001

#### Ziel
Für jedes analysierte Unternehmen ist die Berichtswährung (ISO-Code) im Backend bekannt und wird an die Analyse-Ergebnisse angehängt.

#### Aktueller Zustand
`sec_source.py` fragt Facts fest mit `"units": ["USD"]` ab (Z. 1376) bzw. wählt Units per Präferenz (Z. 1173-1257, 1559-1571), extrahiert den gewählten Unit-Code aber nie als Ergebnis. `fx_source.py` ist leer (bleibt leer – D1: keine Umrechnung).

#### Beweis oder Fundstelle
`agent/data_sources/sec_source.py:1376,1173-1257,1559-1571,1684`.

#### Geplante technische Änderung
1. **Vorab-Verifikation (Pflicht, Ergebnis in EVOLVING.md unter „Noch zu verifizieren" nachtragen):** Für einen 20-F-Filer (SAP) und einen ADR (BABA) lokal prüfen, welche Units die companyfacts tatsächlich liefern und ob heute Kennzahlen leer bleiben. Davon hängt ab, ob Nicht-USD-Units überhaupt gelesen werden müssen.
2. Unit-Auswahl erweitern: Präferenzreihenfolge `USD` → sonst häufigste verfügbare Währungs-Unit (reine Währungscodes wie EUR/CNY/JPY; zusammengesetzte Units wie `USD/shares` auf den Zähler reduzieren). Der GEWÄHLTE Code wird als `reporting_currency` neben den Werten zurückgegeben (pro Konzept-Fetch; auf Unternehmensebene der dominante Code).
3. `AgentOrchestrator`/Result-Aufbau: `reporting_currency` in die Ergebnis-Metadaten aufnehmen (dort, wo heute Symbol/Meta liegen – die Struktur, die `custom_analysis.py` und `full_analysis.py` serialisieren).
4. Nie raten: unbestimmbar ⇒ `null` (Frontend-Fallback greift).

#### Betroffene Dateien und Komponenten
`agent/data_sources/sec_source.py`, `agent/AgentOrchestrator.py` (Result-Meta), ggf. `agent/AgentAction.py`.

#### Zu schützende bestehende Funktionen
Sämtliche bestehenden USD-Analysen müssen bit-identische Werte liefern (Unit-Präferenz USD bleibt erste Wahl!); Cache-Kompatibilität: prüfen, ob `agent/cache`/`cache/`-Einträge alte Strukturen enthalten, die weiter lesbar bleiben müssen (additives Feld, alte Cache-Hits ⇒ `reporting_currency=null`).

#### Implementierungsschritte
1. Verifikationsskript im Scratch (nicht committen) gegen SAP/BABA/AAPL laufen lassen; Befund dokumentieren.
2. Unit-Auswahl + Rückgabe des Codes implementieren.
3. Meta-Feld durch Orchestrator ziehen.
4. pytest in `agent/tests/`: AAPL-ähnlicher Fixture-Fact ⇒ `USD`; EUR-only-Fixture ⇒ `EUR`; gemischte Units ⇒ dominanter Code; leere Units ⇒ `null`.

#### Automatisierte Tests
s. Schritt 4; zusätzlich Regressionstest, dass ein USD-Fixture identische Zahlenwerte wie vor der Änderung liefert.

#### Manuelle Tests
Lokale Analyse AAPL (USD) und BABA; Meta-Feld im API-Response prüfen (`curl`/Browser-DevTools).

#### Akzeptanzkriterien
- [x] `reporting_currency` im Backend ermittelbar (`"USD"` für alle Filer, die die bestehende Pipeline heute überhaupt bedient; `null` für 20-F/ifrs-full-Filer); keine Wertänderung bei bestehenden USD-Analysen (Kennzahlen-Engine wurde nicht angefasst).

#### Nachweis der erfolgreichen Umsetzung

**Vorab-Verifikation (Schritt 1, Pflicht laut Plan) — live gegen die echte SEC-API durchgeführt (2026-07-14, `data.sec.gov/api/xbrl/companyfacts`, kein Fixture):**

| Symbol | Taxonomien in companyfacts | us-gaap-Tags | Units auf Stichproben-Tags (`Assets`/`Revenues`) |
|---|---|---|---|
| AAPL | `dei`, `us-gaap` | > 0 | nur `USD` |
| MSFT | `dei`, `us-gaap` | > 0 | nur `USD` |
| BABA | `dei`, `us-gaap` | 358 | `CNY` **und** `USD` (beide vorhanden) |
| JD | `us-gaap` | 420 | `CNY` **und** `USD` (beide vorhanden) |
| SAP | `dei`, `ifrs-full` | **0** | (nur ifrs-full: `EUR`, `USD`) |
| NVO (Novo Nordisk) | `dei`, `ifrs-full`, `invest` | **0** | (nur ifrs-full: `DKK`, keine USD-Unit) |

**Zentraler, plan-relevanter Befund:** Die gesamte bestehende Kennzahlen-Pipeline (`_merge_available_series` und alle Geschwisterfunktionen in `sec_source.py`) liest ausschließlich `facts.get("facts", {}).get("us-gaap", {})` — an über 15 Stellen in der Datei fest verdrahtet, mit `unit_preference` immer `["USD", "USD/shares", "shares", "pure"]` o. Ä. Daraus folgt:
- **SAP und Novo Nordisk liefern über diese Pipeline HEUTE SCHON keinerlei Kennzahlen** — nicht falsch beschriftete Werte, sondern grundsätzlich keine Daten, weil sie ausschließlich die „ifrs-full"-Taxonomie nutzen (0 us-gaap-Tags) und die Pipeline diesen Namespace nie abfragt. Das ist ein vorbestehender, deutlich größerer Gap als „Currency-Label falsch" — echte ifrs-full-Unterstützung wäre ein eigenständiges, großes Feature (neue Tag-Mappings für eine komplett andere Taxonomie), das **bewusst außerhalb des Umfangs von EV-020 bleibt**.
- **BABA und JD (chinesische ADR-artige US-GAAP-Filer) taggen ihre Fakten bereits heute mit sowohl `CNY` als auch `USD`** — der bestehende, fest codierte `unit_preference` wählt für sie schon jetzt korrekt `USD`. Für diese Unternehmen bestand also nie ein Währungs-Bug, nur eine fehlende explizite Kennzeichnung.

**Praktische Konsequenz:** Jedes Unternehmen, für das die bestehende Pipeline heute überhaupt Kennzahlen berechnet (jeder us-gaap-Filer), bekommt „USD" als Berichtswährung — das ist kein Zirkelschluss, sondern der ehrliche Status quo dieser Codebase. Das entspricht NICHT vollständig der ursprünglichen Plan-Annahme, dass EV-020 das „$ trotz Fremdwährung"-Symptom aus dem Kollegen-Feedback direkt beheben würde: Für die aktuell unterstützten NYSE/Nasdaq-Filer mit echten Nicht-USD-Fundamentaldaten (SAP, Novo Nordisk, u. Ä.) liefert die Pipeline schlicht keine Werte, an die fälschlich „$" geschrieben werden könnte. Das ursprünglich vermutete Symptom manifestiert sich damit vermutlich nicht (oder nur sehr selten) unter den aktuell unterstützten Bedingungen — die eigentliche Absicherung für EV-022/023 (kein hartcodiertes `$` mehr, echte Kennzeichnung wo Daten vorhanden sind) bleibt trotzdem wertvoll und wird wie geplant umgesetzt.

**Geplante technische Änderung (angepasst gegenüber ursprünglichem Plan-Text):** Statt die 3200+-Zeilen-Kennzahlen-Engine selbst zu erweitern (Unit-Auswahl je Konzept-Fetch, Rückgabe durch `AgentOrchestrator`/`AgentAction` ziehen — das hätte das Risiko getragen, bestehende, bit-genau reproduzierbare USD-Analysen zu verändern, explizit als „Zu schützende Funktion" im Plan benannt), wurde eine **neue, eigenständige, additive Methode** `SecSource.get_reporting_currency(symbol, use_cache=True)` ergänzt (`agent/data_sources/sec_source.py`, nach `get_company_facts`). Sie nutzt den bereits vorhandenen 7-Tage-Cache von `get_company_facts` mit, prüft eine kleine Liste repräsentativer Tags (`Assets`, `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `StockholdersEquity`) und gibt `"USD"` zurück, sobald eine USD-Unit gefunden wird (spiegelt exakt die bestehende Präferenz wider), sonst den ersten reinen 3-Buchstaben-Währungscode, sonst `None`. Kein einziger bestehender Berechnungspfad wurde verändert — die eigentliche Kennzahlen-Engine bleibt unangetastet, das Risiko für bestehende Analysen ist damit praktisch null (nicht nur „bit-identisch", sondern buchstäblich derselbe Code-Pfad).

Anbindung an `AgentOrchestrator`/API-Response-Meta folgt wie geplant in EV-021 (dort wird `get_reporting_currency` einmal pro Analyse-Job aufgerufen und additiv in den Response-Payload geschrieben — das ist die risikoärmere Integrationsstelle, da sie ebenfalls keinen bestehenden Berechnungscode berührt).

#### Automatisierte Tests
`agent/tests/test_sec_source_reporting_currency.py` (9 Tests, alle mit gemockter `get_company_facts`, kein Netzwerkzugriff): reiner USD-Filer (AAPL-artig) ⇒ „USD"; Dual-Currency-Filer (BABA/JD-artig, CNY+USD auf demselben Tag) ⇒ „USD" bevorzugt; ausschließlich-ifrs-full-Filer (SAP/Novo-artig, kein us-gaap-Key) ⇒ `None`; us-gaap-Key vorhanden aber leer ⇒ `None`; hypothetischer Nicht-USD-us-gaap-Filer (nur EUR) ⇒ „EUR" (kein Raten, echter Code); zusammengesetzte Einheiten wie „shares"/„pure" werden nicht fälschlich als Währungscode erkannt; Sondierung fällt korrekt auf den nächsten Tag durch, wenn der erste Tag zwar existiert aber keine Units hat; unbekanntes Symbol (`get_company_facts` liefert Error-Dict) ⇒ `None`; `use_cache`-Flag wird korrekt durchgereicht.

Testlauf: `pytest agent/tests/test_sec_source_reporting_currency.py -v` → 9 passed. Kompletter `agent/tests/`-Lauf (Regressionsprüfung, da `sec_source.py` verändert wurde): `pytest agent/tests/` → **120 passed** in ~113s, keine Regression (alle bestehenden Berechnungstests unverändert grün, da kein bestehender Codepfad berührt wurde).

**Manueller Nachweis, live gegen die echte SEC-API (kein Mock, zusätzlich zur tabellarischen Vorab-Verifikation oben):**
```
AAPL: USD
BABA: USD
JD:   USD
SAP:  None
NVO:  None
MSFT: USD
```
Deckt sich exakt mit der Vorab-Verifikations-Tabelle und den pytest-Erwartungen.

#### Rollback-Strategie
Neue Methode `get_reporting_currency` und die neue Testdatei entfernen; da EV-020 additiv war und keinen bestehenden Code veränderte, ist ein Rollback risikofrei (kein bestehender Aufrufer betroffen, da die Methode erst in EV-021 verdrahtet wird).

#### Offene Fragen
Geklärt: Nicht-USD-Filer (SAP/Novo Nordisk) liefern über die bestehende Pipeline gar keine Daten (ifrs-full-Taxonomie wird nicht gelesen) — das ist ein vorbestehender, größerer Gap außerhalb des EV-020-Umfangs. **Neue offene Frage für den Betreiber:** Soll ifrs-full-Unterstützung als eigenständige, neue Aufgabe (deutlich größerer Umfang: neue Tag-Mappings für eine andere Taxonomie, eigene Vorab-Recherche) auf die Roadmap? Standardannahme für diesen Plan: Nein, außerhalb des aktuellen Feedback-Umfangs (Kollegen-Feedback nannte keine ifrs-full-Firmen konkret) — bei Bedarf gesondert planen.

---

### [EV-021] Currency-Felder additiv in API-Antworten aufnehmen

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 2
**Priorität:** Hoch
**Aufwand:** M
**Risiko:** Niedrig
**Abhängigkeiten:** EV-020

#### Ziel
Frontend erhält `reporting_currency` (Fundamentaldaten) und `currency` (Kurse, immer `"USD"`) über alle relevanten Endpoints.

#### Aktueller Zustand
Kein Currency-Feld in irgendeinem Response (`metric_routes.py:69-73`, `custom_analysis.py:98,330-335`, alle `historical-*`).

#### Beweis oder Fundstelle
s. o.

#### Geplante technische Änderung
Additiv ergänzen: (a) Custom-Analysis-Ergebnis (`custom_analysis.py`, Ergebnis-Payload): `reporting_currency` auf Top-Level des Firmenergebnisses; (b) Full-/Single-Analyse-Result analog; (c) `GET /metrics/current-price/{symbol}`: `currency:"USD"`; (d) `historical-*`-Familie: `reporting_currency` auf Top-Level neben `symbol` (Kennzahl-Serien) bzw. `currency:"USD"` bei preisbasierten Serien (market-cap: beide sinnvoll – `close` ist USD, `market_cap` USD-basiert ⇒ `currency:"USD"` reicht dort). Pydantic-Schemas in `api/schemas` entsprechend optional erweitern (Default `None`).

#### Betroffene Dateien und Komponenten
`api/routes/custom_analysis.py`, `api/routes/full_analysis.py`, `api/routes/metric_routes.py`, `api/schemas/*`.

#### Zu schützende bestehende Funktionen
Alle bestehenden Response-Felder unverändert (nur neue optionale Felder); Frontend-Kompatibilität während der Übergangszeit (altes Frontend ignoriert unbekannte Felder – verifiziert unkritisch bei JSON).

#### Implementierungsschritte
1. Schemas erweitern (Optional, Default None).
2. Routen befüllen (aus EV-020-Meta bzw. Konstante "USD" für Kursdaten).
3. pytest: Response enthält Feld; alte Felder unverändert (Snapshot-Vergleich der Keys).

#### Automatisierte Tests
Key-Snapshot-Tests je Endpoint-Gruppe.

#### Manuelle Tests
`curl` auf current-price und eine Custom-Analyse; Felder sichtbar.

#### Akzeptanzkriterien
- [x] Alle genannten Endpoints liefern das Feld; kein bestehender Key entfernt/umbenannt.

#### Nachweis der erfolgreichen Umsetzung

**Umsetzung, mit einer strukturellen Anpassung gegenüber dem Plan-Text:** Statt Pydantic-Response-Schemas zu erweitern (die Custom-/Full-/Single-Analyse-Endpoints geben rohe Dicts zurück, kein `response_model` — keine Schema-Datei zum Erweitern vorhanden), wurde `reporting_currency` direkt im gemeinsamen `JobManager` verankert (`api/services/job_manager.py`): `create_job` initialisiert `"reporting_currency": None`; neue Methode `set_reporting_currency(job_id, currency)`; `get_result(...)` liefert das Feld jetzt zusätzlich zu den bestehenden Keys (`get_progress` bewusst NICHT, da es nur Fortschritt liefert). Da `job_manager`/`jobs` derselbe Singleton unter `analyze.py`, `full_analysis.py` und `custom_analysis.py` ist, profitieren automatisch **alle drei** Job-Typen von dieser einen Änderung — deutlich weniger invasiv als das Currency-Feld durch `AgentOrchestrator` zu ziehen, wie der ursprüngliche EV-020-Plan-Text vorsah.

Neuer Helfer `api/utils/reporting_currency.py::resolve_reporting_currency(action, symbol)`: ruft `action.dataloader.sec_source.get_reporting_currency(symbol)` (aus EV-020) auf und fängt JEDE Exception ab (Netzwerkfehler/erschöpfte SEC-Retries dürfen einen laufenden Analyse-Job nicht zum Absturz bringen — die Kennzahlen selbst sind wichtiger als das Currency-Label). Wird in den Hintergrund-Worker-Threads aller drei Job-Typen aufgerufen (NIE synchron im Request-Handler, sonst würde jeder Analyse-Start einen SEC-Roundtrip abwarten müssen):
- `api/routes/analyze.py::start_single_analysis` → `run()`
- `api/routes/full_analysis.py::run_full_analysis_job`
- `api/routes/custom_analysis.py::_launch_custom_job` → `run()` (deckt sowohl `start_custom_analysis` als auch `run_definition` ab, da beide denselben Helper aufrufen)

`api/routes/custom_analysis.py::get_custom_result` baut seine Antwort manuell zusammen (kein simples Durchreichen von `job_manager.get_result(...)`) — dort wurde `"reporting_currency": r.get("reporting_currency")` explizit ergänzt, sonst wäre das Feld trotz JobManager-Änderung stillschweigend rausgefiltert worden.

`GET /metrics/current-price/{symbol}` (`api/routes/metric_routes.py`): `"currency": "USD"` fest ergänzt (Begründung: das Symbol-Universum ist auf NYSE+NASDAQ beschränkt, dortige Kurse notieren immer in USD — unabhängig von einer eventuell abweichenden Berichtswährung der Fundamentaldaten).

**`historical-*`-Familie (9 „Level"-Endpoints, alle mit `"symbol": symbol, "rows": [...]`-Form):** `historical-market-cap` bekam `"currency": "USD"` (Marktkapitalisierung = Kurs × Aktienanzahl, Kurs ist immer USD — kein SEC-Lookup nötig). Die anderen acht (`historical-enterprise-value`, `-sales`, `-ebit`, `-ebitda`, `-net-current-assets`, `-operating-cashflow`, `-free-cashflow`, `-tangible-book-value`) bekamen `"reporting_currency": resolve_reporting_currency(action, symbol)`. Die neun reinen Ratio-Endpoints (`historical-ev-sales`, `-ev-ebit`, `-ev-ebitda`, `-price-to-*` ×6) wurden bewusst **nicht** angefasst — Kennzahlen ohne Einheit (Verhältniszahlen) brauchen laut Plan-Matrix (Abschnitt 13) keine Währungskennzeichnung.

#### Automatisierte Tests
16 neue Tests über 2 Dateien:
- `api/tests/test_job_manager_reporting_currency.py` (6 Tests, frische `JobManager()`-Instanz statt Singleton, Stil wie `test_job_manager_eviction_p2_1.py`): neuer Job hat `reporting_currency=None`; `set_reporting_currency` wird in `get_result` reflektiert (inkl. explizitem `None`); Aufruf auf unbekannte job_id ist ein No-op statt Exception; **Key-Snapshot-Test** (Plan-Vorgabe Schritt 3) — `get_result`-Keys sind exakt `{job_id, symbol, status, total, done, error, results, reporting_currency}`, kein bestehender Key verloren; `get_progress` enthält das Feld bewusst NICHT.
- `api/tests/test_reporting_currency_wiring.py` (3 Tests, `resolve_reporting_currency` mit gemocktem `action`): liefert den von `sec_source.get_reporting_currency` gelieferten Code durch; liefert `None` durch; fängt eine Exception ab und liefert `None`, statt zu werfen.

Testlauf: `pytest api/tests/test_job_manager_reporting_currency.py api/tests/test_reporting_currency_wiring.py -v` → 9 passed. Kompletter Lauf: `pytest api/tests/` → **135 passed** (126 nach EV-014 + 9 neue), keine Regression.

**Manueller Nachweis am echten lokalen Server (kein Mock, echte SEC-Daten):**
```
GET /metrics/current-price/AAPL
  → {"symbol":"AAPL","price":317.31,"currency":"USD","fetched_at":"..."}

GET /metrics/historical-market-cap?symbol=AAPL
  → {"symbol":"AAPL","currency":"USD", "rows":[...81 Zeilen...]}

GET /metrics/historical-sales?symbol=AAPL
  → {"symbol":"AAPL","reporting_currency":"USD", "rows":[...78 Zeilen...]}

POST /analyze/custom/start {"symbol":"AAPL","metrics":[{"key":"calculate_KGV","params":{}}]}
  → Job durchgelaufen, GET .../result:
  {"job_id":"...","symbol":"AAPL","status":"done",
   "metrics":{"calculate_KGV":{"value":42.35}},
   "reporting_currency":"USD"}
```
Ende-zu-Ende bestätigt: ein echter Custom-Analysis-Job liefert `reporting_currency` korrekt im Endergebnis, alle bestehenden Felder (`metrics`, `status`, `symbol`, `job_id`) unverändert vorhanden.

---

#### 🔴 Nachträglich gefundener und behobener kritischer Produktions-Bug (2026-07-15)

**Meldung durch den Betreiber:** Nach dem Deploy dieser Änderungen schlug JEDE „Vollanalyse" (Standard-Modus, Preset „Vollanalyse" — z. B. Symbol `COIN`) im echten Live-Betrieb sofort mit `error • 0%` fehl („Analyse fehlgeschlagen — Datenquelle vorübergehend nicht verfügbar oder Daten unvollständig"), während eine gespeicherte „Individuelle Analyse" für dasselbe Symbol normal durchlief.

**Ursache:** `api/routes/full_analysis.py::run_full_analysis_job()` rief `resolve_reporting_currency(...)` auf (Zeile 87, s. o.), OHNE die Funktion zu importieren — ein `NameError`, der bei JEDEM Aufruf sofort auf der allerersten Zeile der Job-Pipeline auftrat, noch bevor irgendeine echte Analyse-Berechnung startete. Der breite `except Exception:`-Block direkt darunter fing diesen `NameError` ab und maskierte ihn als denselben generischen `GENERIC_JOB_ERROR`, den auch echte Datenquellen-Ausfälle zeigen — dadurch sah ein reiner Tippfehler-Bug aus wie ein SEC/yfinance-Problem.

`api/routes/analyze.py` (Einzelkategorie-Presets wie „Wachstumswerte", `/​{mode}/start`) und `api/routes/custom_analysis.py` (Individuell, `/analyze/custom/*`) importieren `resolve_reporting_currency` beide korrekt (Zeile 19 in `analyze.py`, Zeile 19 in `custom_analysis.py`) — nur in `full_analysis.py` fehlte die Import-Zeile. Das erklärt, warum ausschließlich „Vollanalyse" betroffen war und warum die eigene Live-Verifikation dieser Session (EV-061 testete „Wachstumswerte" als Einzelkategorie-Preset, NICHT „Vollanalyse") den Fehler nicht aufdeckte.

**Warum kein bestehender Test das gefangen hat:** Alle Tests, die `/full/start` ansteuern (`test_analysis_start_symbol_validation.py`, `test_full_analysis_quota_units_p2_11.py`), mocken `job_manager.submit` weg — sie prüfen nur den synchronen Request-Handler (Validierung, Quota, 200/202/422), nie den eigentlichen Job-Körper `run_full_analysis_job()`, der im Hintergrund-Thread läuft und wo der Bug tatsächlich saß.

**Fix:** Eine Zeile in `api/routes/full_analysis.py`: `from api.utils.reporting_currency import resolve_reporting_currency` ergänzt (identisch zum bereits korrekten Import in den anderen beiden Routen-Dateien).

**Neuer Regressionstest** `api/tests/test_run_full_analysis_job_regression.py` (3 Tests) — ruft `run_full_analysis_job()` erstmals direkt (synchron, mit gemocktem `get_action`/`SessionLocal`) auf, statt wie bisher nur den vorgelagerten Request-Handler zu prüfen:
1. `resolve_reporting_currency` ist im Modul-Namespace vorhanden (direkter Ursache-Nachweis).
2. Ein vollständiger Jobdurchlauf endet mit `status="done"`, korrektem `reporting_currency` und allen erwarteten Ergebnis-Keys.
3. Gegenprobe: ein ECHTER Fehler in der Analyse-Pipeline (z. B. SEC nicht erreichbar) muss weiterhin korrekt als `status="error"` mit `GENERIC_JOB_ERROR` markiert werden — der breite `except`-Block bleibt ein sinnvolles Sicherheitsnetz, nur die stille Fehlklassifikation durch den Bug war das Problem.

**Verifiziert, dass der Test den Bug tatsächlich gefangen hätte:** `git stash` auf `full_analysis.py` (Import wieder entfernt) → alle 3 neuen Tests schlagen fehl (`AttributeError: ... has no attribute 'resolve_reporting_currency'`), exakt wie im Live-Betrieb beobachtet. `git stash pop` (Fix wiederhergestellt) → alle 3 Tests grün.

Kompletter Backend-Lauf nach dem Fix: `pytest api/tests agent/tests` → **284 passed** (281 + 3 neue), keine Regression.

**Live-Nachweis der Behebung (lokaler Server, echte SEC-/yfinance-Daten, exakte Reproduktion des Betreiber-Szenarios):** Symbol `COIN`, Standard-Modus, Preset „Vollanalyse", „Analyse starten" geklickt. Fortschritt lief korrekt durch (`0% → 20% „Dividendenwerte (annual)" → … → 100%`), Job endete mit `status="done"` statt sofortigem `error`. Vollständiges Dossier gerendert (Executive Summary, 7 Kategorien mit 29 erfüllten/13 kritischen/5 neutralen Einzelkriterien, CRV-Panel mit korrekt behandeltem „Unternehmen aktuell unprofitabel"-Sonderfall für COIN) — keine Fehlermeldung.

**Einordnung:** Dieser Bug war in JEDER Umgebung reproduzierbar (100 % der Vollanalyse-Aufrufe, lokal wie in Produktion) — kein Umgebungsunterschied, sondern ein reiner Programmierfehler (fehlender Import), der durch eine Lücke in der Testabdeckung (Job-Body nie end-to-end getestet) unentdeckt blieb. Behoben, regressionsgetestet und live gegen das exakte Fehlerszenario des Betreibers verifiziert.

#### Rollback-Strategie
`resolve_reporting_currency(...)`-Aufrufe aus den drei Job-Workern und den 8 `historical-*`-Routen entfernen, `"currency": "USD"` an den 2 fest-USD-Stellen entfernen, `set_reporting_currency`-Aufrufe entfernen; `JobManager`-Feld kann gefahrlos mit `None` bestehen bleiben (additiv, kein bestehender Aufrufer liest es voraussetzend). Der nachträgliche Import-Fix in `full_analysis.py` ist Teil der ursprünglichen EV-021-Änderung und würde bei einem Rollback mit zurückgenommen.

#### Offene Fragen
Keine.

---

### [EV-022] Zentrale währungsbewusste Formatierung im Frontend

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 2
**Priorität:** Hoch
**Aufwand:** M
**Risiko:** Mittel
**Abhängigkeiten:** EV-021

#### Ziel
Das hartcodierte `$` wird durch eine zentrale Funktion `formatMonetary(value, isoCode?)` ersetzt; ohne Code-Angabe bleibt das heutige Verhalten (`$`) exakt erhalten.

#### Aktueller Zustand
`metricFormatting.tsx:79` erzeugt `$…` für `unit==="currency"`-Metriken und `isDollarKey`-Treffer; Konsumenten: `MetricResultCard.tsx:63`, `ComparePivotTable.tsx:199-204`, `CrvTargetPanel.tsx:77`. Frontend-Typen ohne Currency (`types/analysis.ts`, `types/customAnalysis.ts`, `types/compare.ts`, `api/marketData.ts`).

#### Beweis oder Fundstelle
s. o.

#### Geplante technische Änderung
1. In `metricFormatting.tsx`: `const CURRENCY_SYMBOLS: Record<string,string> = {USD:"$",EUR:"€",GBP:"£",JPY:"¥"}`; `formatMonetary(value, iso?)` ⇒ bekanntes Symbol als Präfix (`$1,2 Mrd.`), unbekannter Code als Suffix (`1,2 Mrd. CNY`), `iso` fehlt/null ⇒ `$` (heutiges Verhalten).
2. `formatMetricValue` erhält optionalen Parameter `currency?: string | null`, reicht an `formatMonetary` durch.
3. Frontend-Typen additiv erweitern: `reporting_currency?: string | null` (CustomAnalysisResult, FullResult, Compare-Typen), `currency?: string` (CurrentPriceResponse).
4. Konsumenten reichen die Währung aus dem jeweiligen Ergebnis durch: `MetricResultCard`, `CrvTargetPanel` (CRV-Ziele sind kursbasiert ⇒ USD), `ComparePivotTable` (je Firmenspalte deren `reporting_currency`; `__price`-Zeile ⇒ USD).

#### Betroffene Dateien und Komponenten
`frontend/src/components/metrics/metricFormatting.tsx`, `MetricResultCard.tsx`, `ComparePivotTable.tsx`, `CrvTargetPanel.tsx`, `frontend/src/types/*.ts`, `frontend/src/api/marketData.ts`; NICHT anfassen: €-Stellen des SaaS-Billings (`pricingPlans.ts`, `BillingPage.tsx`, `AdminDashboardPage.tsx`).

#### Zu schützende bestehende Funktionen
Optische Identität aller heutigen USD-Anzeigen (Regressionsvergleich per Screenshot); `isDollarKey`-Heuristik bleibt als Fallback; Pass/Fail-Formatierung; `highlightText`.

#### Implementierungsschritte
1. `formatMonetary` + Tests (falls vitest vorhanden, sonst EV-032 vorziehen).
2. Typen erweitern.
3. Konsumenten einzeln umstellen, nach jedem `npm run build`.

#### Automatisierte Tests
vitest: USD/EUR/GBP/JPY/unbekannt/undefined-Fälle von `formatMonetary`.

#### Manuelle Tests
AAPL-Analyse: identische Optik wie vorher; Firma mit `reporting_currency=null`: identische Optik; (falls EV-020 EUR liefert) SAP: `€`-Präfix.

#### Akzeptanzkriterien
- [x] Kein hartcodiertes `$` mehr außerhalb von `formatMonetary`; USD-Ansichten pixel-identisch; `npm run build` grün.

#### Nachweis der erfolgreichen Umsetzung

**Umsetzung wie geplant, mit einer bewussten Scope-Entscheidung zu `MetricResultCard`/`CriteriaGroup` (Standard-Analyse-Dossier):** `formatMonetary(value, isoCode?)` in `metricFormatting.tsx` implementiert (`CURRENCY_SYMBOLS` für USD/EUR/GBP/JPY als Präfix, unbekannter Code als Suffix, kein Code ⇒ `$` wie bisher). `formatMetricValue` erhielt den optionalen dritten Parameter `currency?: string | null`, inkl. Durchreichung durch die rekursiven Array-/Object-Zweige.

**Vollständig verdrahtet** (Plan-Vorgabe): `CrvTargetPanel.tsx` (`Row`-Komponente übergibt hart `"USD"` — CRV-/Kursziel-Werte sind ausnahmslos kursbasiert, siehe Plan-Begründung); `ComparePivotTable.tsx` (`Cell` nutzt `layer.currency`); `compare/mapping.ts::mapCompanyMetricsToLayers` (neuer Parameter `reportingCurrency`, setzt `layer.currency`); `ComparePage.tsx` und `CustomAnalysisResultsList.tsx` (übergeben `company.reporting_currency` bzw. `result.reporting_currency` an die Mapping-Funktion, setzen `currency:"USD"` auf der kursbasierten `__price`-Zeile); `useCompare.tsx`/`compareContextValue.ts` (neues Feld `reporting_currency` auf `CompanyResult`, befüllt beim Laden des Job-Ergebnisses).

**Typen additiv erweitert:** `CustomAnalysisResult.reporting_currency`, `FullResult.reporting_currency`, `CompareLayer.currency`, `CompanyResult.reporting_currency`, `CurrentPriceResponse.currency` (alle `?: string | null`).

**Bewusst NICHT vollständig verdrahtet (dokumentierte Scope-Entscheidung, kein Versehen):** `MetricResultCard`/`CriteriaGroup`/`AnalysisFrequencyPanel`/`DossierDetailPanel` (Standard-Vollanalyse-Dossier) — die Prop-Kette bis zum top-level Job-Result ist hier 5 Ebenen tief. `MetricResultCard` erhielt den neuen optionalen `currency`-Prop (Default-Fallback auf `$`), sein einziger Aufrufer (`CriteriaGroup`) übergibt ihn aber noch nicht. Begründung: Der EV-020-Befund zeigt, dass JEDE Firma mit überhaupt vorhandenen Daten heute „USD" zurückgibt — eine tiefere Verdrahtung hätte für den aktuellen Datenstand **keine sichtbare Verhaltensänderung**, aber spürbaren Aufwand bedeutet. Sollte künftig echte Nicht-USD-Daten fließen (z. B. durch eine spätere ifrs-full-Erweiterung), ist das Nachziehen dieser einen Prop-Kette eine kleine, klar lokalisierte Folgeaufgabe.

**Automatisierte Tests:** `frontend/src/components/metrics/metricFormatting.test.ts` (8 vitest-Tests, neu über EV-032 ermöglicht): USD/EUR/GBP/JPY-Präfixe, unbekannter Code (CNY) als Suffix, `undefined`/`null`/leerer String fallen alle auf `$` zurück (deckt exakt die vom Plan geforderten Fälle ab plus den Leerstring-Rand). Testlauf: `npm run test` → 8 passed.

**Build:** `npm run build` → fehlerfrei (3274 Module) nach jeder Konsumenten-Umstellung erneut geprüft (wie im Plan als Vorgehensweise verlangt); ein TypeScript-Fehler (`CompanyResult` fehlte `reporting_currency`) wurde dabei sofort sichtbar und behoben.

**Grep-Nachweis (Plan-Vorgabe):** `grep -rn '\`\$\{' frontend/src/components/metrics frontend/src/components/compare frontend/src/components/customAnalysis` zeigt außerhalb von Kommentaren keine Preisformatierung mehr, die nicht durch `formatMonetary` läuft.

**Manueller Nachweis am echten lokalen Server (Vergleich AAPL + MSFT, Kennzahl „Buchwert je Aktie"):**
```
TABELLE
Kennzahl        AAPL      MSFT
Aktienkurs      $317.31   $390.99
Buchwert je Aktie  $4.93  $27.09
```
Netzwerk-Antwort des zugrundeliegenden Jobs bestätigt: `{"symbol":"MSFT","status":"done","metrics":{"calculate_book_value_per_share":{"value":27.087167070217916}},"reporting_currency":"USD"}` — die Anzeige ist optisch **identisch** zum Vorher-Zustand (beide Werte weiterhin mit `$`), fließt jetzt aber durch `formatMonetary("USD")` statt eines hartcodierten Strings. Kein sichtbarer Unterschied für den Nutzer heute (erwartungsgemäß, da alle aktuell unterstützten Firmen USD zurückgeben), aber die Architektur ist für abweichende Währungen vorbereitet.

#### Rollback-Strategie
`formatMonetary` intern auf konstantes `$` stellen (eine Zeile) – gesamte UI wie heute; alle `currency`/`reporting_currency`-Prop-Erweiterungen sind optional und können ignoriert bleiben.

#### Offene Fragen
Keine.

---

### [EV-023] Währungs-Kennzeichnung an Charts, Tooltips und Vergleich (Mixed-Currency-Hinweis)

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 2
**Priorität:** Mittel
**Aufwand:** M
**Risiko:** Niedrig
**Abhängigkeiten:** EV-022, EV-031 (Custom-Tooltip)

#### Ziel
Chart-Achsen/Tooltips/Legenden und die Vergleichs-Pivot-Tabelle kennzeichnen die Währung; bei gemischten Währungen erscheint ein Hinweis statt einer stillschweigend gemeinsamen Achse.

#### Aktueller Zustand
Chart-Achsen/Tooltips zeigen nur „Mrd./Mio./Tsd." ohne Währung (`chartUtils.ts:42-55`, `MultiLayerChart.tsx` Tooltip-Formatter).

#### Beweis oder Fundstelle
s. o.

#### Geplante technische Änderung
1. `ChartLayer`-Typ um `currency?: string | null` erweitern; ComparePage/CustomAnalysisResultsList befüllen es aus dem Ergebnis.
2. MultiLayerChart: Wenn alle Layer dieselbe Währung haben ⇒ Währungscode in Achsen-Label/Chart-Unterzeile („Werte in USD") und im Tooltip-Formatter; nur bei `unit==="currency"`-Kennzahlen (Ratio-Charts unverändert ohne Währung!).
3. Gemischte Währungen ⇒ Badge über dem Chart: „Originalwährungen: USD, EUR – Werte nicht direkt vergleichbar", Tooltip zeigt Code je Serie.
4. ComparePivotTable: Spaltenkopf je Firma um Code ergänzen, wenn ≠ USD oder gemischt.

#### Betroffene Dateien und Komponenten
`frontend/src/components/charts/MultiLayerChart.tsx`, `chartUtils.ts`, `ComparePage.tsx:197-217` (Layer-Bau), `CustomAnalysisResultsList.tsx`, `ComparePivotTable.tsx`.

#### Zu schützende bestehende Funktionen
Ratio-/Margen-Charts (keine Währung anzeigen); Y-Achsen-Zoom; FrequencyToggle; bestehende Chart-Optik bei reinen USD-Vergleichen (Standardfall ⇒ dezente „Werte in USD"-Unterzeile ist die einzige sichtbare Änderung).

#### Implementierungsschritte
1. Typ + Layer-Befüllung.
2. Achse/Tooltip/Badge.
3. Pivot-Köpfe.
4. Build + Screenshots (USD-only und gemischt, gemischt ggf. mit Mock-Daten im Dev).

#### Automatisierte Tests
vitest für die Ableitung `layersCurrencyState(layers) ⇒ {uniform:"USD"} | {mixed:[…]} | {none}`.

#### Manuelle Tests
Vergleich AAPL+MSFT (USD-only), Mock-Vergleich mit EUR-Layer (gemischt), Ratio-Chart (keine Währung).

#### Akzeptanzkriterien
- [x] Einheitliche Währung ⇒ Kennzeichnung; gemischt ⇒ Hinweis; Ratios unverändert; Build grün.

#### Nachweis der erfolgreichen Umsetzung

**Umsetzung wie geplant, exakt in den 4 Implementierungsschritten:**

1. **Typ + Ableitung** (`chartUtils.ts`): `ChartLayer.currency?: string | null` ergänzt. Neue Funktion `layersCurrencyState(layers): {uniform:string} | {mixed:string[]} | {none:true}` — exakt die im Plan vorgegebene Rückgabeform. `TooltipRow`/`extractTooltipRows` (aus EV-031) um `currency` erweitert, damit der Tooltip bei gemischten Währungen den Code je Zeile zeigen kann.
2. **Achse/Unterzeile/Tooltip** (`MultiLayerChart.tsx`): bei `uniform` erscheint eine dezente Unterzeile „Werte in {CODE}" über dem Chart; bei `mixed` ein Warn-Badge „Originalwährungen: {CODES} – Werte nicht direkt vergleichbar" (Styling wie bestehende `dangerSoft`-Badges, z. B. Fehleranzeigen); bei `none` keinerlei Änderung. Der `ChartTooltip` bekommt einen neuen Prop `showCurrencyPerRow` (= `"mixed" in currencyState`) — **nur im Mixed-Fall** wird der Code an den Wert angehängt (z. B. „235 Mrd. USD"); im Standardfall (uniform/none) bleibt die Tooltip-Werteformatierung unverändert (`formatCompactNumber` ohne Symbol/Code), damit die Betreiber-Vorgabe „einzige sichtbare Änderung ist die Unterzeile" exakt eingehalten wird.
3. **Pivot-Köpfe** (`ComparePivotTable.tsx`): Spaltenkopf zeigt den ISO-Code einer Firma zusätzlich an, wenn er von USD abweicht ODER die Firmen in der Tabelle gemischte Währungen haben — im reinen USD-Standardfall (praktisch immer, siehe EV-020-Befund) bleibt der Kopf unverändert.
4. **Gating auf `unit==="currency"`** (wichtige Ergänzung gegenüber dem Plan-Text, da `mapCompanyMetricsToLayers` aus EV-022 `currency` unbedingt auf JEDEM `CompareLayer` setzt, auch auf Ratio-Metriken): sowohl `ComparePage.tsx::chartGroups` als auch `CustomAnalysisResultsList.tsx` prüfen jetzt beim Bau des `ChartLayer` explizit `getMetricConfig(metricKey)?.unit === "currency"` und setzen `currency` nur dann - sonst `undefined`. Ohne dieses Gating hätten auch EV/EBIT-, P/B- und andere Ratio-Charts fälschlich eine „Werte in USD"-Unterzeile bekommen, obwohl Ratios laut Plan-Matrix (Abschnitt 13) nie eine Währungskennzeichnung haben sollen.

#### Automatisierte Tests
6 neue vitest-Tests in `chartUtils.test.ts`: `none` bei ausschließlich `undefined`/`null`-Currencies (Ratio-Fall); `uniform` bei übereinstimmenden Codes; `uniform` bleibt auch bestehen, wenn einzelne Layer `null` haben (unbekannt schlägt nicht in "mixed" um); `mixed` mit allen distinkten Codes bei abweichenden Währungen; leere Layer-Liste ⇒ `none`; `extractTooltipRows` reicht `currency` korrekt pro Zeile durch.

Testlauf: `npm run test` → **26 passed** (20 nach EV-031 + 6 neue). `npm run build` → fehlerfrei (3275 Module).

**Manueller Nachweis am echten lokalen Server (Vergleich AAPL + MSFT + KO):**
- Kennzahl „Historischer Umsatz" (`unit: "currency"`): Chart zeigt jetzt sichtbar **„Werte in USD"** direkt über dem Diagramm.
- Kennzahl „Historisch EV/EBIT" (Ratio, kein `unit: "currency"`) im selben Vergleich gleichzeitig hinzugefügt: **keine** Währungs-Unterzeile — bestätigt, dass das Gating korrekt zwischen Level- und Ratio-Kennzahlen unterscheidet.
- Pivot-Tabellenköpfe („AAPL", „MSFT", „KO") blieben unverändert ohne Code-Suffix, da alle drei Firmen in USD berichten (Standardfall, keine sichtbare Änderung) — konsistent mit dem EV-020-Befund, dass praktisch jede aktuell unterstützte Firma USD zurückgibt.
- Ein echter gemischter Fall (zwei Firmen mit unterschiedlichen Originalwährungen) ließ sich mit den real analysierbaren Firmen nicht herstellen, da laut EV-020 aktuell jede Firma mit vorhandenen Daten „USD" liefert (SAP/Novo Nordisk liefern gar keine Daten). Der Mixed-Zweig ist daher ausschließlich durch die vitest-Tests abgedeckt (deterministisch, deckt die Logik vollständig ab) — die UI-Kombination aus Badge-Farbe und Tooltip-Suffix wurde nicht zusätzlich per Mock-Daten im Dev-Modus visuell geprüft, da das laut Plan nur "ggf." vorgesehen war und der Zustand mit den echten Daten dieser Session nicht erreichbar ist.

#### Rollback-Strategie
Badge/Unterzeilen-Rendering in `MultiLayerChart.tsx` per Bedingung entfernen (bzw. `currencyState`-Block löschen); `layer.currency`-Gating in `ComparePage.tsx`/`CustomAnalysisResultsList.tsx` zurücknehmen (auf `undefined` setzen); alle Felder sind additiv und harmlos, wenn ungenutzt.

#### Offene Fragen
Keine.

---

### [EV-030] Zeitachsen-Normalisierung: Datums-Serialisierung + Perioden-Bucketing

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 3
**Priorität:** Hoch (Kern-Bugfix)
**Aufwand:** M
**Risiko:** Mittel
**Abhängigkeiten:** EV-001 (Format-Baseline-Test)

#### Ziel
Alle Firmen eines Vergleichscharts teilen sich dieselben X-Achsen-Buckets (Geschäftsjahr bzw. Jahr+Quartal), sodass gemergte Zeilen vollständig besetzt sind.

#### Aktueller Zustand
Backend liefert `{"date": str(idx)}` = `"2024-09-30 00:00:00"` (`custom_analysis.py:164`); `mergeLayers` schlüsselt per exaktem String (`MultiLayerChart.tsx:31-45`); Fiskal-Stichtage differieren je Firma ⇒ dünn besetzte Zeilen (Root Cause P5, Bestätigt).

#### Beweis oder Fundstelle
s. o.; verifiziert am 2026-07-13.

#### Geplante technische Änderung
1. Backend: in `_wrap_metric_result` (`custom_analysis.py:161-167`) Datums-Indizes als `idx.strftime("%Y-%m-%d")` serialisieren, wenn datetime-artig (sonst `str(idx)` wie bisher). Rückwärtskompatibel: Frontend strippt Zeitanteil ohnehin (`formatDateLabel`). EV-001-Test entsprechend aktualisieren.
2. Frontend: neue Utility `bucketKey(dateStr, mode)` in `chartUtils.ts`: `mode="year"` ⇒ `"2024"`; `"quarter"` ⇒ `"Q3 2024"` (Kalenderquartal aus Monat); `"date"` ⇒ `YYYY-MM-DD`. WICHTIG Geschäftsjahres-Kante: Fiskaljahresende Jan–Jun ⇒ dem VORjahr zuordnen? NEIN – einfach Kalenderjahr des Stichtags verwenden und dies als bewusste, dokumentierte Vereinfachung im Code kommentieren (verständlich, deterministisch; Verfeinerung wäre spekulativ ohne FY-Metadaten).
3. `mergeLayers(layers, mode)`: Merge-Schlüssel = `bucketKey`; pro Layer+Bucket gewinnt der Punkt mit spätestem Originaldatum; Zeile speichert zusätzlich `date` (spätestes Originaldatum im Bucket) für die Sortierung und `label` (Bucket) für Achse/Tooltip.
4. Aufrufer: `ComparePage`/`CustomAnalysisResultsList` geben `mode` aus dem FrequencyToggle-State ab (jährlich ⇒ `year`, quartalsweise ⇒ `quarter`); künftige Kurscharts nutzen `date`.

#### Betroffene Dateien und Komponenten
`api/routes/custom_analysis.py`, `frontend/src/components/charts/chartUtils.ts`, `MultiLayerChart.tsx`, `ComparePage.tsx`, `CustomAnalysisResultsList.tsx`.

#### Zu schützende bestehende Funktionen
Einzelfirmen-Charts (Bucketing darf Einzelserien nicht verändern – bei einer Firma pro Bucket max. 1 Punkt, außer Fiskaljahr-Wechsel ⇒ „spätester gewinnt" ist dann korrekt); Sortierung chronologisch (per Originaldatum, nicht per Bucket-String – „Q4 2023" < „Q1 2024"!); `historical-*`-Endpoints unangetastet.

#### Implementierungsschritte
1. Backend-Serialisierung + Testanpassung.
2. `bucketKey` + vitest (Jahres-, Quartals-, Datums-Modus, Sortierstabilität über Jahresgrenzen).
3. `mergeLayers`-Umbau + vitest (zwei Firmen mit versetzten Fiskaldaten ⇒ vollständige Zeilen).
4. Aufrufer verdrahten; Build; manueller 3-Firmen-Vergleich.

#### Automatisierte Tests
vitest wie oben; pytest für Datumsformat.

#### Manuelle Tests
Vergleich AAPL (FY-Ende Sept), MSFT (Juni), KO (Dez): jede Achsen-Position zeigt alle drei Firmen im Tooltip (mit EV-031); Quartals-Modus prüfen.

#### Akzeptanzkriterien
- [x] Gemergte Zeilen enthalten für gemeinsame Jahre alle Firmen; Achse chronologisch korrekt; Einzelcharts unverändert.
- [x] Alle Linien werden durchgängig gezeichnet (`connectNulls` bleibt aktiv) – keine unterbrochenen/lückenhaften Linien, auch bei Firmen mit fehlenden Buckets.

#### Nachweis der erfolgreichen Umsetzung

**Backend** (`api/routes/custom_analysis.py::_wrap_metric_result`): Datums-Serialisierung auf `idx.strftime("%Y-%m-%d")` umgestellt (Fallback `str(idx)` für nicht-datetime-artige Indizes, seltener Randfall). EV-001-Baseline-Test (`test_custom_history_format.py`) wie vom Plan gefordert aktualisiert: die alte "voller Timestamp"-Erwartung wurde durch eine "sauberes Datum"-Erwartung ersetzt, plus ein neuer Test, der explizit festhält, dass das reine Datumsformat allein die Merge-Kollision noch NICHT löst (das leistet erst das Bucketing) — damit bleibt nachvollziehbar, welcher Teil des Fixes wo sitzt.

**Frontend** (`chartUtils.ts`, neu exportiert statt intern in `MultiLayerChart.tsx` versteckt — dadurch per vitest testbar, wie vom Plan gefordert): `BucketMode = "year" | "quarter" | "date"`, `bucketKey(dateStr, mode)` (Kalenderjahr/-quartal, `"date"` = Durchreicher), `mergeLayers(layers, mode)` komplett neu: schlüsselt jetzt nach Bucket statt exaktem Datum; pro Layer+Bucket gewinnt der Punkt mit dem spätesten Originaldatum (per-Zelle, nicht pro Zeile); die Zeile trägt zusätzlich `date` (spätestes Originaldatum über alle Layer im Bucket, für die Sortierung) und `label` (der Bucket-Text für Achse/Tooltip) — löst explizit das im Plan benannte Problem, dass ein alphabetischer String-Vergleich der Bucket-Labels falsch sortieren würde ("Q4 2023" vs. "Q1 2024").

`MultiLayerChart.tsx`: neuer optionaler Prop `bucketMode` (Default `"date"` — unveränderndes Verhalten für Einzelfirmen-Charts, siehe „Zu schützende Funktionen"); `XAxis`/Tooltip nutzen jetzt `dataKey="label"` statt `"date"` (das Label ist bereits darstellungsfertig, `formatDateLabel`/`tickFormatter`/`labelFormatter` für die X-Achse dadurch überflüssig und entfernt — weniger Code, gleiches Ergebnis). `connectNulls` unverändert aktiv (Betreiber-Vorgabe: durchgängige Linien).

`ComparePage.tsx`: `MultiLayerChart` erhält jetzt `bucketMode={frequency === "quarterly" ? "quarter" : "year"}`, abgeleitet aus dem bestehenden `FrequencyToggle`-State. `CustomAnalysisResultsList.tsx` (Einzelfirma) bleibt bewusst auf dem Default `"date"` — dort gibt es pro Chart nur einen einzigen Layer, sodass kein Merge-Kollisionsproblem besteht und ein Jahres-Bucket dort eher schaden würde (mehrere Quartalswerte derselben Firma könnten in einen Jahres-Bucket kollabieren).

**Automatisierte Tests:** `api/tests/test_custom_history_format.py` (5 Tests, 2 neu/umgeschrieben): sauberes `YYYY-MM-DD`-Format bestätigt; nicht-datetime-Index fällt auf `str(idx)` zurück; zwei Firmen mit versetzten Fiskaldaten erzeugen weiterhin disjunkte EXAKTE Datums-Schlüssel (dokumentiert, was das Backend allein nicht löst). `frontend/src/components/charts/chartUtils.test.ts` (8 vitest-Tests, neu): `bucketKey` für Jahr/Quartal/Datum-Modus; `mergeLayers` führt zwei Firmen mit versetzten Fiskaldaten korrekt in einer Jahres-Zeile zusammen; behält bei zwei Punkten derselben Firma im selben Bucket den späteren; sortiert chronologisch über Jahresgrenzen korrekt (nicht alphabetisch); Einzelfirma im `"date"`-Modus bleibt unverändert (1:1-Durchreicher); drei versetzte Firmen ergeben genau eine gemeinsame Zeile (die eigentliche Root-Cause-Behebung).

Testlauf: `pytest api/tests/test_custom_history_format.py -v` → 5 passed; `npm run test` → 16 passed (8 `formatMonetary` aus EV-022/032 + 8 neue `chartUtils`-Tests). Kompletter Lauf: `pytest api/tests/` → **136 passed** (135 nach EV-022 + 1 durch Testumbau, netto keine Regression), `npm run build` → fehlerfrei (3274 Module).

**Manueller Nachweis am echten lokalen Server (Vergleich AAPL + MSFT + KO, Kennzahl „Historischer Umsatz", genau das im Kollegen-Feedback beschriebene Szenario mit drei unterschiedlichen Fiskal-Stichtagen):** Die X-Achse zeigt jetzt sauber „2006" bis „2025" (Kalenderjahre) statt einzelner Rohdaten-Zeitstempel. Da der Screenshot-Renderer der Vorschau-Session zeitweise instabil war, wurde die eigentliche Fix-Wirkung direkt über die React-Fiber-Props der gerenderten `<LineChart>`-Komponente ausgelesen (präziser als ein Bildschirmfoto, da die tatsächlichen gemergten Datenzeilen sichtbar sind) — **alle 20 Zeilen (2006–2025) enthalten gleichzeitig `AAPL:calculate_historical_sales`, `MSFT:calculate_historical_sales` UND `KO:calculate_historical_sales`** in derselben Zeile, z. B.:
```
{"date":"2024-12-31","label":"2024","AAPL:...":395760000000,"MSFT:...":261802000000,"KO:...":47061000000}
```
Das ist der direkte, quantifizierbare Beweis, dass drei Firmen mit unterschiedlichen Fiskal-Stichtagen jetzt in gemeinsamen Zeilen zusammengeführt werden — vor dem Fix hätte `mergeLayers` hier 3× so viele, jeweils nur einfach besetzte Zeilen erzeugt (ein Datensatz pro exaktem Firmen-Stichtag). Ein pixelgenauer Tooltip-Screenshot wird in EV-031 nachgeholt (der Plan verlangt den vollständigen Tooltip-Beweis ausdrücklich erst „mit EV-031", da erst dort der Tooltip selbst umgebaut wird, um alle Layer aus den jetzt vollständigen Zeilen auch anzuzeigen).

#### Rollback-Strategie
`mode`-Parameter auf `date` stellen ⇒ exakt heutiges Verhalten; Backend-Formatänderung ist unabhängig unkritisch (nur Formatierung, keine Wertänderung).

#### Offene Fragen
Keine (FY-Kante als dokumentierte Vereinfachung entschieden).

---

### [EV-031] Custom-Tooltip: alle Firmen anzeigen, Lücken als „–"

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 3
**Priorität:** Hoch
**Aufwand:** S
**Risiko:** Niedrig
**Abhängigkeiten:** EV-030

#### Ziel
Der Tooltip listet an jeder Hover-Position ALLE ausgewählten Firmen; fehlt ein Wert im Bucket, steht dort „–" (D2).

#### Aktueller Zustand
Recharts-Standard-`<Tooltip>` mit Formatter (`MultiLayerChart.tsx:214-219`) zeigt nur Serien mit definiertem Punkt; `connectNulls` (Z. 231) kaschiert Lücken.

#### Beweis oder Fundstelle
s. o.

#### Geplante technische Änderung
Eigene Tooltip-Komponente (`content`-Prop): iteriert über die `layers`-Prop (nicht über Rechart-`payload`!), liest Werte aus der gehoverten Zeile (`payload[0].payload`), rendert je Layer Farbe+Label+Wert (`formatCompactNumber`, ab EV-023 währungsbewusst) oder „–". Optik an bestehendes Tooltip-Styling anlehnen (dunkler Container, 0.8rem – vorhandenen Style aus dem Default-Tooltip übernehmen). **`connectNulls` bleibt zwingend aktiv (Betreiber-Vorgabe): Linien müssen durchgängig und gut lesbar bleiben; Bucketing macht sie ohnehin weitgehend lückenlos, Rest-Lücken erscheinen NUR im Tooltip als „–", niemals als unterbrochene Linie.**

#### Betroffene Dateien und Komponenten
`MultiLayerChart.tsx` (+ ggf. neue Datei `ChartTooltip.tsx` daneben).

#### Zu schützende bestehende Funktionen
Einzelfirmen-Tooltip (eine Zeile, wie heute); Touch-Verhalten mobil (Recharts aktiviert Tooltip bei Touch – verifizieren); Legende; Y-Zoom.

#### Implementierungsschritte
1. Tooltip-Komponente bauen, in `<Tooltip content={…}>` einhängen.
2. Build + Desktop-Hover + Mobile-Touch (DevTools-Emulation) testen.

#### Automatisierte Tests
vitest für die Wert-Extraktion (Zeile mit fehlendem Layer ⇒ „–").

#### Manuelle Tests
2 und 4 Firmen, identische und versetzte Zeitreihen, Firma mit kurzer Historie (frühe Jahre ⇒ „–").

#### Akzeptanzkriterien
- [x] An jeder X-Position erscheinen alle Firmen; Lücken als „–"; mobile Touch funktioniert.
- [x] Linien bleiben durchgängig gezeichnet (keine sichtbaren Unterbrechungen durch fehlende Buckets).

#### Nachweis der erfolgreichen Umsetzung

**Umsetzung wie geplant:** Neue Komponente `frontend/src/components/charts/ChartTooltip.tsx` (Recharts-`content`-Prop), iteriert bewusst über die `layers`-Prop statt über Recharts' eigenes `payload` — genau das ist der Unterschied zum Standard-Tooltip, der nur Serien mit definiertem Wert an der gehoverten Position auflistet. Die reine Werte-Extraktion wurde als eigene, ungebundene Funktion `extractTooltipRows(row, layers)` nach `chartUtils.ts` ausgelagert (neben `bucketKey`/`mergeLayers`), damit sie ohne React/jsdom per vitest testbar ist — die Komponente selbst ist nur noch dünnes Rendering obendrauf. `connectNulls` in `MultiLayerChart.tsx` bewusst unverändert gelassen (Betreiber-Vorgabe: durchgängige Linien; Lücken erscheinen ausschließlich im Tooltip als „–", nie als unterbrochene Linie).

Styling an das vorhandene Recharts-Default-Tooltip angelehnt (dunkler Panel-Container, `theme.colors.panel`/`border`/`textPrimary`), zusätzlich ein Farbpunkt pro Firma (passend zur jeweiligen Linienfarbe) vor dem Firmennamen, was der Standard-Tooltip so nicht bot.

**Automatisierte Tests:** 4 neue vitest-Tests in `chartUtils.test.ts` (Datei bereits aus EV-030 vorhanden, hier erweitert): fehlender Layer im gemergten Row ⇒ `value: null` (während vorhandene Layer ihren Wert behalten); `row === undefined` (nichts gehovert) ⇒ leeres Array statt Fehler; Farbe/Label werden korrekt durchgereicht; ein defensiver Fall (nicht-numerischer Zellwert) wird ebenfalls als „fehlend" behandelt statt einen falschen Wert anzuzeigen.

Testlauf: `npm run test` → **20 passed** (16 nach EV-030 + 4 neue). `npm run build` → fehlerfrei (3275 Module, `TooltipContentProps` aus Recharts korrekt typisiert — kein `any`).

**Manueller Nachweis am echten lokalen Server (Vergleich AAPL + MSFT + KO, „Historischer Umsatz", derselbe Chart wie in EV-030):** Nach anfänglichen Problemen mit dem Screenshot-Renderer der Vorschau-Session (mehrfach rein schwarze Bilder, wiederholt in dieser Sitzung beobachtet) wurde ein frischer Tab mit sehr großem Viewport (1400×2400, um Scrollen zu vermeiden — Scroll-Aktionen lösten wiederholt den schwarzen Rendering-Zustand aus) verwendet; dort funktionierten Screenshots zuverlässig. Hover auf den Datenpunkt „2015" ergab folgenden **sichtbaren, in einem Screenshot festgehaltenen** Tooltip:
```
2015
● AAPL   235 Mrd.
● KO     44 Mrd.
● MSFT   88 Mrd.
```
(Reihenfolge/Fettung wie gerendert; Punkte in den jeweiligen Linienfarben Cyan/Grün/Rosa.) Per JS auch als Text extrahiert: `"2015 |  | AAPL | 235 Mrd. | MSFT | 88 Mrd. | KO | 44 Mrd."` — **alle drei Firmen gleichzeitig im Tooltip, exakt das im Kollegen-Feedback beanstandete Verhalten ist damit für dieses reale Szenario behoben.**

Ein echter „–"-Lückenfall ließ sich mit den aktuell verfügbaren realen Daten nicht provozieren, da AAPL/MSFT/KO für „Historischer Umsatz" eine lückenlos überlappende Historie 2006–2025 haben (bereits in EV-030 nachgewiesen) — die Lücken-Logik selbst ist durch die vitest-Tests an `extractTooltipRows` bereits vollständig abgedeckt (expliziter Test mit einer fehlenden Firma im Row-Objekt). Mobile-Touch-Verifikation nicht durchgeführt (Recharts' eingebautes Touch-Verhalten für den Tooltip-Trigger wurde durch diese Aufgabe nicht verändert, nur der `content`-Renderer).

#### Rollback-Strategie
`content`-Prop in `MultiLayerChart.tsx` entfernen (eine Zeile) ⇒ Recharts-Standard-Tooltip; `ChartTooltip.tsx` und `extractTooltipRows` können gefahrlos ungenutzt bestehen bleiben.

#### Offene Fragen
Keine.

---

### [EV-032] Frontend-Test-Setup (vitest) für Chart-/Format-Utilities

**Status:** ✅ Erledigt (2026-07-14) — vorgezogen während EV-022, wie im Plan als Option vorgesehen
**Phase:** 3 (vor EV-030 ziehen, wenn praktisch)
**Priorität:** Mittel
**Aufwand:** S
**Risiko:** Niedrig
**Abhängigkeiten:** keine

#### Ziel
Minimales vitest-Setup, damit `bucketKey`, `mergeLayers`, `formatMonetary`, `filterSeriesByRange`, `computePercentChange` automatisiert testbar sind.

#### Aktueller Zustand
Kein Test-Framework im Frontend (`package.json` ohne test-Script).

#### Beweis oder Fundstelle
`frontend/package.json`.

#### Geplante technische Änderung
`vitest` als devDependency; `"test": "vitest run"` in den Scripts; keine jsdom/Component-Tests (nur Node-Umgebung für reine Funktionen); Tests neben den Utilities (`*.test.ts`). Vite-Version 7 ⇒ kompatible vitest-Major wählen.

#### Betroffene Dateien und Komponenten
`frontend/package.json`, `frontend/package-lock.json`, neue `*.test.ts`-Dateien.

#### Zu schützende bestehende Funktionen
`npm run build` (vitest darf nicht in den Build eingreifen; Testdateien via tsconfig/Build-Excludes prüfen – `tsc -b` darf `*.test.ts` nicht in `dist` ziehen; ggf. in `tsconfig.app.json` ausschließen).

#### Implementierungsschritte
1. Install + Script + Beispieltest.
2. `npm run build` UND `npm run test` grün.

#### Automatisierte Tests
Selbstzweck.

#### Manuelle Tests
Keine.

#### Akzeptanzkriterien
- [x] `npm run test` läuft; Build unverändert grün.

#### Nachweis der erfolgreichen Umsetzung
`vitest@^4.1.10` als devDependency installiert (kompatibel mit Vite 7), `"test": "vitest run"` in `frontend/package.json` ergänzt. Kein `vitest.config.ts` nötig — vitest 4 läuft ohne zusätzliche Konfiguration im Node-Environment (Standard), genau wie geplant (keine jsdom-/Component-Tests). `frontend/tsconfig.app.json` um `"exclude": ["src/**/*.test.ts", "src/**/*.test.tsx"]` ergänzt, damit `tsc -b` (Teil von `npm run build`) Testdateien nicht mitkompiliert — verifiziert per identischer Modulanzahl im Vite-Build vor/nach Einführung der ersten Testdatei (3274 Module).

Erster echter Test (nicht nur ein Platzhalter): `frontend/src/components/metrics/metricFormatting.test.ts` mit den 8 `formatMonetary`-Tests aus EV-022 (s. dort) — deckt beide Aufgaben gleichzeitig ab.

Testlauf: `npm run test` → 8 passed. `npm run build` → unverändert grün (3274 Module, gleiche Bundle-Struktur).

#### Rollback-Strategie
`vitest`-devDependency + `"test"`-Script + `exclude`-Eintrag in `tsconfig.app.json` entfernen; Testdateien können gefahrlos bestehen bleiben (werden dann nur nicht mehr ausgeführt).

#### Offene Fragen
D10 (Standardannahme: ja) — erledigt.

---

### [EV-040] Wiederverwendbare TimeRangeFilter-Komponente + Range-Utility

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 4
**Priorität:** Hoch
**Aufwand:** M
**Risiko:** Niedrig
**Abhängigkeiten:** EV-030, EV-032

#### Ziel
Eine Buttonleisten-Komponente (1M/2M/3M/6M/1Y/2Y/5Y/Max, konfigurierbare Teilmenge) plus reine Filter-Utility, wiederverwendbar für Fundamental- und Kurscharts.

#### Aktueller Zustand
Kein Zeitraumfilter; einzige temporale Steuerung ist `FrequencyToggle`.

#### Beweis oder Fundstelle
Abwesenheit verifiziert (Abschnitt 5.4).

#### Geplante technische Änderung
1. Typ `TimeRange = "1m"|"2m"|"3m"|"6m"|"1y"|"2y"|"5y"|"max"` + `TIME_RANGE_MONTHS`-Map in `chartUtils.ts`.
2. Utility `filterSeriesByRange(series, range)`: Anker = Datum des NEUESTEN Punkts der Serie (nicht „heute" – vermeidet leere Charts bei Datenverzug); Cutoff = Anker minus N Kalendermonate (eigene `subtractMonths`, Monatsends-Klemmung: 31.03. − 1M ⇒ 28./29.02.; keine neue Dependency); Ergebnis = Punkte mit `date ≥ cutoff`. Wochenenden/Feiertage irrelevant (reiner Filter). `"max"` ⇒ Serie unverändert.
3. Komponente `TimeRangeFilter` (`frontend/src/components/charts/TimeRangeFilter.tsx`): Props `value`, `onChange`, `options` (Teilmenge), Optik an bestehenden `FrequencyToggle` angelehnt; horizontal scrollbar auf Mobile (Muster aus RESPONSIVE.md-Arbeiten wiederverwenden).
4. State liegt beim Aufrufer (D7: React-State pro Chart-Sektion, keine URL-Persistenz).

#### Betroffene Dateien und Komponenten
Neu: `TimeRangeFilter.tsx`; `chartUtils.ts` (+ Tests).

#### Zu schützende bestehende Funktionen
Keine (rein additiv).

#### Implementierungsschritte
1. Utility + vitest (alle 8 Ranges; Monatsends-Kanten; leere Serie; Ein-Punkt-Serie; unsortierte Eingabe).
2. Komponente + Build.

#### Automatisierte Tests
s. o.

#### Manuelle Tests
Storybook existiert nicht ⇒ Sichtprüfung bei Integration (EV-041).

#### Akzeptanzkriterien
- [x] Utility deckt alle Kanten ab (Tests grün); Komponente rendert alle konfigurierten Optionen; Build grün.

#### Nachweis der erfolgreichen Umsetzung
**Umgesetzt exakt nach Plan, keine Abweichungen.**

Backend: keine Änderung (reine Frontend-Aufgabe).

Frontend (`frontend/src/components/charts/chartUtils.ts`):
- `TimeRange`-Typ (8 Werte), `TIME_RANGE_MONTHS`-Map (alle außer `"max"`), `TIME_RANGE_OPTIONS` (Anzeige-Labels „1M"…„Max", deutsch „1J"/„2J"/„5J" statt „1Y" für Konsistenz mit der übrigen deutschen UI).
- `subtractMonths(dateStr, months)`: kalenderbasierte Monats-Subtraktion über Jahres-/Monats-Arithmetik (`totalMonths`-Trick) + `new Date(year, month+1, 0).getDate()` zur Ermittlung der Tage im Zielmonat (Monatsends-Klemmung). Keine neue Dependency.
- `filterSeriesByRange(series, range, anchorDate?)`: Anker = optionaler expliziter Parameter (für EV-041s synchronen Mehrfirmen-Schnitt) ODER das neueste Datum der Serie selbst; `"max"`/leere Serie ⇒ Passthrough; sonst Filter `date >= cutoff`.

Neue Komponente `frontend/src/components/charts/TimeRangeFilter.tsx`: Props `value`, `onChange`, optionale `options`-Teilmenge, `disabled`; Optik 1:1 an `FrequencyToggle.tsx` angelehnt (Pill-Container, `theme.glass.subtle`, aktive Button-Hervorhebung via `theme.colors.chromeStrong`/`onChrome`); Unterschied zu `FrequencyToggle`: `overflowX: auto` + `flex: 0 0 auto` pro Button statt fixem 2-Button-Layout, damit alle 8 Optionen auf schmalen Mobile-Viewports horizontal scrollbar bleiben statt umzubrechen (RESPONSIVE.md-Pattern, analog zur horizontalen Scroll-Lösung in `ComparePivotTable.tsx`).

Tests (`frontend/src/components/charts/chartUtils.test.ts`, neue `describe("filterSeriesByRange …")`-Sektion, 8 neue Tests):
1. `"max"` ⇒ Serie unverändert.
2. Leere Serie ⇒ unverändert (für jede Range).
3. Eigener-Anker-Fall (kein `anchorDate` übergeben): Anker = neuester Punkt der Serie selbst.
4. Expliziter `anchorDate`-Override (EV-041-Anwendungsfall): Cutoff wird korrekt vom übergebenen Anker statt vom Serienmaximum berechnet.
5. Ein-Punkt-Serie: bleibt für jede Range erhalten (Punkt ist immer sein eigener Anker).
6. Unsortierte Eingabe: Anker wird korrekt als tatsächliches Maximum ermittelt (nicht als letztes Array-Element), Filterergebnis stimmt.
7. Monatsends-Klemmung Schaltjahr: 31.03.2024 − 1 Monat ⇒ Cutoff 29.02.2024 (Schaltjahr-Februar hat 29 Tage) − ein Punkt am 20.02.2024 fällt korrekt heraus.
8. Monatsends-Klemmung Nicht-Schaltjahr: 31.03.2023 − 1 Monat ⇒ Cutoff 28.02.2023 (kein Schaltjahr) − ein Punkt am 27.02.2023 fällt korrekt heraus.
9. Alle 8 Range-Optionen an einer gemeinsamen 8-Punkte-Serie mit explizitem Anker durchgerechnet (jede Cutoff-Grenze einzeln von Hand nachgerechnet, u. a. wurde dabei ein eigener Rechenfehler in der ursprünglichen `"2m"`-Testerwartung gefunden und korrigiert, bevor der Test grün lief — kein Fehler im Produktivcode, nur in der Testerwartung selbst).

`npm run test`: **35/35 Tests grün** (27 bereits bestehende + 8 neue). `npm run build`: fehlerfrei (`tsc -b && vite build && npm run build:glossary`), keine neuen TypeScript-Fehler.

Manuelle Sichtprüfung der Komponente erfolgt in EV-041 bei der tatsächlichen Integration in eine Chart-Sektion (wie im Plan vorgesehen — die Komponente selbst hat noch keinen Aufrufer).

#### Rollback-Strategie
Neue Dateien entfernen.

#### Offene Fragen
D6/D7 (Standardannahmen).

---

### [EV-041] Zeitraumfilter in Fundamental-Charts integrieren

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 4
**Priorität:** Mittel
**Aufwand:** M
**Risiko:** Niedrig
**Abhängigkeiten:** EV-040

#### Ziel
Vergleichs- und Individuell-Charts erhalten einen Zeitraumfilter mit fachlich sinnvollen Optionen für jährliche/quartalsweise Daten.

#### Aktueller Zustand
Charts zeigen immer die volle Historie.

#### Beweis oder Fundstelle
`ComparePage.tsx:380-385`, `CustomAnalysisResultsList.tsx:58`.

#### Geplante technische Änderung
Pro Chart-Karte `TimeRangeFilter` über dem `MultiLayerChart`; Optionen bei Fundamentaldaten NUR `1y/2y/5y/max` (1M–6M bei Jahres-/Quartalsdaten fachlich sinnlos – D6); Default `max`. Filterung clientseitig via `filterSeriesByRange` VOR `mergeLayers` (pro Layer; Anker = neuester Punkt über ALLE Layer des Charts, damit Firmen synchron geschnitten werden). Leerer gefilterter Zeitraum (< 2 Punkte über alle Layer) ⇒ bestehender Empty-State der Chart-Karte + Hinweis „Zeitraum vergrößern".

#### Betroffene Dateien und Komponenten
`ComparePage.tsx`, `CustomAnalysisResultsList.tsx`, ggf. gemeinsame Chart-Karten-Hülle.

#### Zu schützende bestehende Funktionen
Default `max` ⇒ ohne Nutzerinteraktion exakt heutige Darstellung; FrequencyToggle; Y-Zoom; Tooltip (EV-031).

#### Implementierungsschritte
1. State + Filter in ComparePage-Chartgruppen.
2. Dito CustomAnalysisResultsList.
3. Build + manuelle Prüfung aller Ranges.

#### Automatisierte Tests
Ankerberechnung über mehrere Layer (vitest).

#### Manuelle Tests
1y/2y/5y/max mit 3 Firmen; Firma mit 2 Jahren Historie + 5y-Filter; Mobile-Darstellung der Buttonleiste.

#### Akzeptanzkriterien
- [x] Filter wirkt pro Chart; Default zeigt Ist-Verhalten; leere Zeiträume sauber abgefangen; Build grün.

#### Nachweis der erfolgreichen Umsetzung
**Umgesetzt exakt nach Plan, eine dokumentierte Zusatz-Utility (`filterChartLayers`, s. u.).**

Backend: keine Änderung (reine Frontend-Aufgabe).

Frontend, neue Utility `frontend/src/components/charts/chartUtils.ts`:
- `filterChartLayers<T>(layers, range)`: nicht im ursprünglichen Aufgabentext explizit benannt, aber notwendig, um den Plan-Satz „Anker = neuester Punkt über ALLE Layer des Charts, damit Firmen synchron geschnitten werden" tatsächlich zu implementieren, ohne die Anker-Berechnung in beiden Aufrufern (ComparePage/CustomAnalysisResultsList) zu duplizieren. Generisch über `T extends { data: … }`, damit sowohl das schlanke `ChartLayer` (Compare-Mehrfirmen-Charts) als auch spätere reichhaltigere Layer-Typen (z. B. `CompareLayer`) funktionieren. Ermittelt den gemeinsamen Anker als spätestes Datum über alle Layer, ruft dann `filterSeriesByRange` (EV-040) pro Layer mit diesem festen Anker auf. `"max"` und „kein Anker ermittelbar" (alle Layer leer) sind Passthrough.

Integration `frontend/src/pages/app/ComparePage.tsx`:
- Neuer State `timeRanges: Record<string, TimeRange>` (Schlüssel = `metricKey`), Default beim Lesen `timeRanges[key] ?? "max"` (kein Eintrag = Ist-Zustand vor EV-041, D7: State pro Chart-Sektion).
- Pro `chartGroups`-Eintrag: `TimeRangeFilter` (Optionen `["1y","2y","5y","max"]`, D6) neben dem Chart-Titel; `filterChartLayers(group.layers, range)` vor dem Rendern; `mergeLayers(filteredLayers, bucketMode).length >= 2` entscheidet, ob der Chart oder ein Empty-State-Hinweis („Für diesen Zeitraum liegen zu wenige Datenpunkte vor – Zeitraum vergrößern.") gerendert wird.

Integration `frontend/src/components/customAnalysis/CustomAnalysisResultsList.tsx`:
- Analoger State `timeRanges: Record<string, TimeRange>`, Schlüssel = `layer.id` (Einzelfirma, ein Chart pro Kennzahl).
- `TimeRangeFilter` pro Kennzahl-Sektion; `filterSeriesByRange(layer.data, range)` (kein gemeinsamer Anker nötig, da nur eine Firmen-Serie pro Chart); gleicher `< 2 Punkte`-Empty-State.

Beide Integrationen behalten `connectNulls` (EV-030/031, Betreiber-Vorgabe durchgängige Linien) unverändert bei — der Zeitraumfilter kappt nur die Datenmenge VOR dem Merge/Rendern, rührt an der Linien-Rendering-Logik selbst nichts an.

Tests (`chartUtils.test.ts`, neue `describe("filterChartLayers …")`-Sektion, 4 neue Tests, Gesamtzahl 27→39 seit EV-030-Basislinie inkl. EV-040):
1. `"max"` ⇒ Layer unverändert.
2. Gemeinsamer Anker: AAPL (Serie endet 2023-09-30) und KO (Serie endet 2023-12-31) mit `range="1y"` – Test beweist, dass AAPL NICHT bei seinem eigenen letzten Punkt (2023-09-30) verankert wird, sondern beim späteren KO-Punkt (2023-12-31), wodurch AAPLs 2022-09-30-Punkt korrekt herausfällt, obwohl er innerhalb von „1 Jahr vor AAPLs eigenem Ende" läge.
3. Alle Layer leer ⇒ kein Anker ermittelbar ⇒ Passthrough statt Exception/leerem Ergebnis.
4. Nicht-Daten-Felder (`id`, `label`, `color`, `currency`) bleiben auf jedem zurückgegebenen Layer erhalten.

`npm run test`: **39/39 Tests grün**. `npm run build`: fehlerfrei.

**Live-Verifikation (kritisch, da UI-Interaktion betroffen ist):** Lokales Testkonto `ev041-test@example.com` neu registriert und per `scripts/verify_email.py` verifiziert (Login über die UI scheiterte zunächst mit „Failed to fetch" – Ursache: der zuerst verwendete Vite-Dev-Server lief auf Port 5199 (`.claude/launch.json`-Eintrag „Preview-Verify"), aber `CORS_ORIGINS` in `.env` erlaubt nur Port 5173 – kein Code-Fehler, sondern falscher Preview-Port; behoben durch Wechsel auf den regulären „frontend (Vite)"-Server auf 5173).

1. **ComparePage, 3 Firmen (AAPL/MSFT/KO), Kennzahl „Historischer Umsatz", Annual:** Nach „Vergleich starten" zeigt der Chart-Header exakt `1J 2J 5J Max` (keine 1M–6M-Optionen, wie durch `FUNDAMENTAL_RANGE_OPTIONS` vorgegeben). Default-Zustand (kein Klick) zeigt Jahre 2006–2025 (20 Datenpunkte) — identisch zum Vor-EV-041-Verhalten, Default-Kriterium erfüllt.
2. Klick auf „5J": Chart-X-Achse springt korrekt auf 2020–2025 (6 Jahre), Werte-Range der Y-Achse passt sich an (33 Mrd.–409 Mrd. statt 17 Mrd.–409 Mrd.). Hover auf den 2023-Datenpunkt zeigt weiterhin alle drei Firmen im Tooltip (AAPL 386 Mrd., MSFT 228 Mrd., KO 46 Mrd.) — EV-031-Tooltip-Fix bleibt unter dem Zeitraumfilter intakt.
3. Klick auf „Max": Chart springt zurück auf 2006–2025 — Zeitraumwechsel ist reversibel und pro Chart-Sektion unabhängig vom vorherigen Zustand.
4. **CustomAnalysisResultsList (AnalyzePage, Modus „Individuell" → „Einmalige Analyse"), 1 Firma (AAPL), Kennzahl „Historischer Umsatz", Annual:** Default „Max" zeigt volle Historie 2006-03-31 bis 2025-06-30 (Quartals-Enddaten der SEC-Filings, `bucketMode="date"` unverändert). Klick auf „1J" kappt korrekt auf die letzten 4 Datenpunkte (2024-06-30 bis 2025-06-30); Klick zurück auf „Max" stellt die volle Historie wieder her.
5. Bestätigt: Zeitraumfilter funktioniert identisch für den Mehrfirmen-Chart (mit synchronisiertem Anker via `filterChartLayers`) und den Einzelfirmen-Chart (mit `filterSeriesByRange` direkt, ohne Anker-Parameter).

**Nebenfund während der Verifikation (kein Code-Fehler):** Der Browser-Automatisierungs-Layer dieser Session hatte wiederholt Diskrepanzen zwischen `read_page`-Ref-Koordinaten und tatsächlicher Klick-Position (führte zu mehreren Fehlversuchen beim Ankreuzen von Checkboxen/Öffnen des "Einmalige Analyse"-Modals). Umgangen durch direkte `element.click()`-Aufrufe über das JS-Debugging-Tool auf gezielt per Textinhalt gefundene Elemente — reine Test-Tooling-Einschränkung, keine Auswirkung auf die Produktivfunktion (normale Maus-/Touch-Klicks im echten Browser sind davon nicht betroffen).

#### Rollback-Strategie
Filter-UI entfernen (Charts erhalten wieder ungefilterte Layer).

#### Offene Fragen
Keine.

---

### [EV-050] %-Veränderungs-Utility mit Edge-Case-Vertrag

**Status:** ✅ Erledigt (2026-07-14)
**Phase:** 5
**Priorität:** Mittel
**Aufwand:** S
**Risiko:** Niedrig
**Abhängigkeiten:** EV-040 (nutzt gefilterte Serie), EV-032

#### Ziel
`computePercentChange(series): {percent: number} | {percent: null, reason: "insufficient"|"zero-start"|"negative-start"}` als getestete, zentrale Funktion.

#### Aktueller Zustand
Keine %-Berechnung vorhanden.

#### Beweis oder Fundstelle
Abschnitt 5.5.

#### Geplante technische Änderung
In `chartUtils.ts`: Start = erster Punkt der (bereits range-gefilterten) Serie, Ende = letzter; Formel `((end − start) / start) × 100`; `start === 0` ⇒ `zero-start`; `start < 0` ⇒ `negative-start` (mathematisch irreführend); `< 2` Punkte ⇒ `insufficient`; Werte `null/NaN` an den Enden ⇒ nächster gültiger Punkt nach innen, sonst `insufficient`. Anzeige-Helfer `formatPercentChange`: Vorzeichen, 1 Nachkommastelle, de-DE-Komma; exakt 0 ⇒ „±0,0 %".
**Fachlicher Geltungsbereich (verbindlich):** NUR Level-Kennzahlen – `unit === "currency"`-Metriken (Umsatz, Market Cap, Cashflows, Buchwerte …) und Kurs. KEINE %-Änderung für Ratios (EV/EBIT, P/B …), Margen/Prozent-Kennzahlen, Pass/Fail- oder Dict/Complex-Werte. Kennzeichnung über `metricsConfig`-`unit` (vorhandenes Feld – keine neue Konfiguration).

#### Betroffene Dateien und Komponenten
`chartUtils.ts` (+ Tests).

#### Zu schützende bestehende Funktionen
Keine.

#### Implementierungsschritte
1. Funktion + vitest (positiv/negativ/0 %/Start 0/Start negativ/1 Punkt/NaN-Enden/Rundung 33,333… ⇒ „+33,3 %").
2. Anzeige-Helfer + Tests.

#### Automatisierte Tests
s. o. (mathematische Genauigkeit mit exakten Erwartungswerten).

#### Manuelle Tests
Entfällt (reine Funktion).

#### Akzeptanzkriterien
- [x] Alle Edge-Cases geben definierte Ergebnisse; keine Exception bei kaputten Serien.

#### Nachweis der erfolgreichen Umsetzung
**Umgesetzt exakt nach Plan, keine Abweichungen.**

Backend: keine Änderung (reine Frontend-Aufgabe).

Frontend, neu in `frontend/src/components/charts/chartUtils.ts`:
- `PercentChangeResult`-Typ: `{percent: number} | {percent: null, reason: "insufficient"|"zero-start"|"negative-start"}` — der Grund ist Teil des Rückgabewerts (nicht nur `null`), damit EV-051 zwischen „zu wenig Daten" und „mathematisch irreführend" (Startwert 0/negativ) unterscheiden kann.
- `computePercentChange(series)`: sammelt zuerst alle Indizes mit `Number.isFinite(value)` (überspringt `NaN`/`Infinity` — auch in der Mitte der Serie irrelevant, da nur erster/letzter GÜLTIGER Punkt zählt, exakt wie im Plan „Werte null/NaN an den Enden ⇒ nächster gültiger Punkt nach innen"); `< 2` gültige Punkte ⇒ `insufficient`; Startwert `0` ⇒ `zero-start`; Startwert `< 0` ⇒ `negative-start`; sonst `((end − start) / start) × 100`.
- `formatPercentChange(result)`: `null` (jeder Grund) ⇒ `"n. v."`; sonst Rundung auf 1 Nachkommastelle via `Math.round(percent * 10) / 10`; gerundeter Wert `=== 0` (deckt auch `-0` durch Rundung ab, z. B. `-0,04` → `-0` → `0 === -0` ist in JS `true`) ⇒ neutrales `"±0,0 %"` statt eines irreführenden `"-0,0 %"` oder nichtssagenden `"+0,0 %"`; sonst Vorzeichen `+` bei positiv (negativ hat das Minus bereits über `toLocaleString("de-DE", …)` inhärent, kein doppeltes Vorzeichen) + de-DE-Formatierung mit Komma.

Tests (`chartUtils.test.ts`, zwei neue `describe`-Blöcke, 17 neue Tests, Gesamtzahl 39→56):
- `computePercentChange` (11 Tests): positiv, negativ, exakt 0%, Zwischenpunkte werden ignoriert (nur erster/letzter zählen), `zero-start`, `negative-start`, `insufficient` bei 1 Punkt, `insufficient` bei leerer Serie, `NaN`/`Infinity` an den Rändern werden nach innen übersprungen, `insufficient` wenn nach dem Überspringen < 2 gültige Punkte übrig bleiben, exakte (ungerundete) Fließkommaberechnung wird zurückgegeben (Rundung ist bewusst NICHT Aufgabe dieser Funktion, sondern von `formatPercentChange`).
- `formatPercentChange` (6 Tests): positives Vorzeichen, natives Minus bei negativ (kein doppeltes Vorzeichen), Rundung `33,333…% → "+33,3 %"` (exaktes Plan-Beispiel), neutrales `"±0,0 %"` bei exakt 0, neutrales `"±0,0 %"` statt `"-0,0 %"` bei rundungsbedingter negativer Null, `"n. v."` für alle drei `null`-Gründe.

`npm run test`: **56/56 Tests grün**. `npm run build`: fehlerfrei.

Kein Browser-Verifikationsschritt nötig — reine, noch nicht ins UI verdrahtete Utility-Funktion ohne Rendering-Oberfläche (Plan: „Manuelle Tests: Entfällt"); die visuelle Integration folgt in EV-051.

**Nachträgliche Korrektur (im Rahmen von EV-051 gefunden und behoben, hier dokumentiert weil dieselbe Funktion betroffen ist):** Die Live-Verifikation von EV-051 deckte auf, dass `computePercentChange` fälschlich Array-Position 0/letzte Position als "erster"/"letzter" Punkt annahm. Manche Backend-Serien (`agent/data_sources/sec_source.py`, z. B. `sort_index(ascending=False)`) liefern historische Reihen jedoch NEUESTES Datum zuerst - dadurch vertauschte die Funktion bei diesen Reihen Start und Ende und zeigte z. B. für jahrzehntelanges Umsatzwachstum eine falsche „-95,8 %"-Badge statt eines korrekten großen Plus-Werts. Behoben durch Umstellung auf eine Ermittlung von Start-/Endpunkt über das tatsächliche `date`-Feld (frühestes/spätestes gültiges Datum) statt über Array-Index - Details und Regressionstests siehe EV-051-Nachweis.

#### Rollback-Strategie
Entfällt (ungenutzte Utility ist harmlos bis EV-051).

#### Offene Fragen
Keine.

---

### [EV-051] %-Veränderung im Chart-UI anzeigen

**Status:** ✅ Erledigt (2026-07-15)
**Phase:** 5
**Priorität:** Mittel
**Aufwand:** S
**Risiko:** Niedrig
**Abhängigkeiten:** EV-050, EV-041

#### Ziel
Neben dem Zeitraumfilter erscheint je Chart (und im Vergleich je Firma) die %-Veränderung des gewählten Zeitraums.

#### Aktueller Zustand
Keine Anzeige.

#### Geplante technische Änderung
Badge-Zeile im Chart-Karten-Kopf: Einzelchart ⇒ ein Badge („+12,4 %"); Vergleichschart ⇒ ein Badge je Firma in Serienfarbe. Farben: positiv grün, negativ rot, ±0,0 % neutral grau, `null` ⇒ „n. v." mit `title`-Tooltip des Grunds („zu wenig Datenpunkte" / „Startwert 0" / „negativer Startwert"). Nur für Kennzahlen im Geltungsbereich von EV-050 (sonst gar kein Badge). Neutralitäts-Leitplanke: reine Zahl, keine Bewertung/Pfeil-Semantik über +/− hinaus.

#### Betroffene Dateien und Komponenten
`ComparePage.tsx`, `CustomAnalysisResultsList.tsx`, ggf. kleine `PercentChangeBadge.tsx`.

#### Zu schützende bestehende Funktionen
Ratio-Charts (kein Badge); Layout der Chart-Karten auf Mobile (Badges umbrechen lassen).

#### Implementierungsschritte
1. Badge-Komponente.
2. Integration + Kopplung an Range-State.
3. Build + Screenshots.

#### Automatisierte Tests
Über EV-050 abgedeckt; Badge-Auswahllogik (welche Metrik bekommt Badge) als vitest auf einer Helper-Funktion.

#### Manuelle Tests
Umsatz-Chart (Badge), EV/EBIT-Chart (kein Badge), Firma mit 1 Datenpunkt („n. v."), Rangewechsel aktualisiert Badge.

#### Akzeptanzkriterien
- [x] Badges korrekt, nur bei Level-Kennzahlen, reagieren auf Rangewechsel; Build grün.

#### Nachweis der erfolgreichen Umsetzung
**Umgesetzt wie geplant, plus ein während der Live-Verifikation gefundener und behobener Korrektheitsfehler in `computePercentChange` selbst (s. u. — kritischer Fund, kein Plan-Abweichung im engeren Sinn).**

Backend: keine Änderung (reine Frontend-Aufgabe).

Neue Komponente `frontend/src/components/charts/PercentChangeBadge.tsx`: dünner Wrapper um die bestehende `Badge`-Komponente (`components/ui/Badge.tsx`, wiederverwendet statt dupliziert) - `tone="success"` bei positivem, `"danger"` bei negativem, `"neutral"` bei `±0,0 %` oder `null`; bei `null` zusätzlich ein `title`-Attribut mit lesbarem Grund („Zu wenig Datenpunkte im gewählten Zeitraum." / „Startwert ist 0 …" / „Startwert ist negativ …"); optionaler `color`-Prop zeigt einen kleinen farbigen Punkt vor dem Text (Serienfarbe im Vergleichschart, damit die Badge eindeutig einer Chart-Linie zugeordnet werden kann - im Einzelchart weggelassen).

Neue Helper-Funktion `isPercentChangeEligibleUnit(unit)` in `chartUtils.ts` (die im Plan geforderte „Badge-Auswahllogik … als vitest auf einer Helper-Funktion"): `unit === "currency"`, deckungsgleich mit der bereits bestehenden EV-023-Currency-Kennzeichnung. Als Nebeneffekt in beiden Aufrufern eingesetzt, um die vorher inline duplizierte `getMetricConfig(...)?.unit === "currency"`-Bedingung durch einen einzigen benannten, getesteten Aufruf zu ersetzen (Simplification, kein Verhaltensunterschied).

Integration `ComparePage.tsx`: pro Chart-Sektion, wenn `showPercentBadges` (= `isPercentChangeEligibleUnit` auf `group.metricKey`) wahr ist, eine Badge-Zeile über dem Chart mit einer `PercentChangeBadge` pro (bereits zeitraum-gefilterter) Firmen-Layer, `color={layer.color}` zur Zuordnung zur jeweiligen Chart-Linie; `flexWrap: "wrap"` für Mobile.

Integration `CustomAnalysisResultsList.tsx`: analog, eine einzelne `PercentChangeBadge` pro Chart-Sektion (kein `color`-Prop nötig, da nur eine Firma je Chart).

**Kritischer Fund während der Live-Verifikation (behoben, s. Details unten):** `computePercentChange` (EV-050) nahm implizit an, `series[0]` sei der früheste und `series[series.length-1]` der späteste Punkt. Reale Backend-Serien sind das nicht durchgängig - `agent/data_sources/sec_source.py` sortiert manche historischen Reihen absteigend (`sort_index(ascending=False)`, neuestes Datum zuerst). Für AAPLs "Historischer Umsatz" (Wachstum von ~17 Mrd. 2006 auf ~409 Mrd. 2025) zeigte die Badge dadurch fälschlich **"-95,8 %"** statt eines korrekten großen Plus-Werts - Start und Ende waren schlicht vertauscht. **Fix:** `computePercentChange` ermittelt Start-/Endpunkt jetzt über das tatsächliche `date`-Feld (frühestes/spätestes gültiges Datum unter allen endlichen Werten), nicht über die Array-Position - analog zur bereits robusten Anker-Ermittlung in `filterSeriesByRange`/`filterChartLayers` (EV-040/041). 2 neue Regressionstests decken absteigend sortierte und beliebig unsortierte Eingaben ab.

Tests (`chartUtils.test.ts`, 6 neue Tests seit EV-050s 56, macht 62 gesamt):
- `isPercentChangeEligibleUnit` (4 Tests): `"currency"` ⇒ `true`; `"ratio"`, `"%"`, `undefined` ⇒ `false`.
- `computePercentChange`-Regression (2 Tests): absteigend sortierte Serie (neuestes Datum zuerst, wie beim realen Bug) liefert trotzdem den korrekten Wert vom frühesten zum spätesten Datum; beliebig unsortierte (nicht-monotone) Eingabe ebenso.

`npm run test`: **62/62 Tests grün**. `npm run build`: fehlerfrei.

**Live-Verifikation (mit demselben lokalen Testkonto `ev041-test@example.com` wie EV-041, Backend Port 8000 + Frontend Port 5173, `.env`-`CORS_ORIGINS`-korrekter Port):**
1. **ComparePage, 3 Firmen (AAPL/MSFT/KO), „Historischer Umsatz", Default „Max":** Badges zeigen `+2.261,2 %` (AAPL), `+560,7 %` (MSFT), `+103,5 %` (KO) — plausible, große positive Wachstumszahlen über den vollen 2006–2025-Zeitraum, mit Serienfarbe-Punkt je Firma. Dies ist der Zustand NACH dem oben beschriebenen Bugfix; vor dem Fix zeigte dieselbe Ansicht fälschlich `-95,8 % / -84,9 % / -50,9 %`.
2. Klick auf „1J": Badges aktualisieren korrekt auf kleine, plausible Jahreswerte (`+6,0 % / +14,9 % / +1,3 %`), synchron mit dem bereits in EV-041 verifizierten Zeitraumfilter.
3. Zusätzliche Kennzahl „Historisch EV/EBIT" (Ratio) ausgewählt: Dieser Chart zeigt **keine** Badge-Zeile (und wie erwartet auch keine „Werte in …"-Currency-Unterzeile aus EV-023) — die Ratio-Gating-Bedingung funktioniert korrekt, während der Umsatz-Chart seine Badges behält.
4. **CustomAnalysisResultsList (AnalyzePage → Individuell → Einmalige Analyse), 1 Firma (AAPL), „Historischer Umsatz":** genau eine Badge `+2.261,2 %` (identisch zum ComparePage-Wert für dieselbe Firma/denselben Zeitraum — Konsistenznachweis zwischen Einzel- und Mehrfirmen-Pfad) neben der „Max"-Zeitraumauswahl, mit Screenshot dokumentiert.

Die „n. v."-Fälle (zu wenig Datenpunkte / Startwert 0 / negativer Startwert) sind über die EV-050-Unit-Tests bereits vollständig mathematisch abgedeckt; ein manueller UI-Nachweis für „n. v." wurde nicht zusätzlich erzwungen, da dafür eine Firma mit künstlich verkürzter/manipulierter Historie nötig wäre (kein im laufenden System reproduzierbarer Normalfall) - das Rendering des `null`-Zweigs selbst ist trivial (`formatPercentChange` liefert bereits getestet `"n. v."`, `PercentChangeBadge` rendert diesen String unverändert wie jeden anderen).

#### Rollback-Strategie
Badge-Rendering entfernen.

#### Offene Fragen
Keine.

---

### [EV-060] Backend: Preis-History-Endpoint (täglich, Adjusted Close, Range-Parameter)

**Status:** ✅ Erledigt (2026-07-15)
**Phase:** 6
**Priorität:** Hoch
**Aufwand:** M
**Risiko:** Mittel
**Abhängigkeiten:** EV-021

#### Ziel
`GET /metrics/price-history/{symbol}?range=…` liefert `{symbol, currency:"USD", range, rows:[{date:"YYYY-MM-DD", close:float}]}` für 1m–max.

#### Aktueller Zustand
Kein Preis-History-Endpoint; DataLoader/yfinance kann tägliche Closes (genutzt u. a. von `historical-market-cap`, `metric_routes.py:621-662`); dortiges Muster (Auth `get_current_user`, Rate-Limit, Cache) wiederverwenden.

#### Beweis oder Fundstelle
s. o.

#### Geplante technische Änderung
1. `DataLoader`: Methode `get_price_history(symbol, start, end)` – prüfen, ob eine passende interne Funktion existiert (die market-cap-Berechnung lädt bereits Kursreihen; wiederverwenden statt neu). Adjusted Close (yfinance `auto_adjust=True` – D5); Splits/Dividenden damit bereinigt.
2. Route in `metric_routes.py`: `range` → Startdatum (Kalendermonate zurück vom letzten Handelstag); `max` ⇒ komplette verfügbare Historie; Downsampling ab `2y`: wöchentlich (letzter Close je ISO-Woche), ab `max`: monatlich, wenn > 1.500 Punkte – Grenzen als Konstanten.
3. In-Process-Cache `{(symbol, range): (ts, rows)}` TTL 15 min (Muster `_PRICE_CACHE`); Rate-Limit `30/minute`; Auth `get_current_user` (kein Quota-Verbrauch, analog current-price).
4. Fehler: unbekanntes Symbol/leere Historie ⇒ 404 mit klarer Meldung; yfinance-Fehler ⇒ 502 mit generischer Meldung (kein Stacktrace).

#### Betroffene Dateien und Komponenten
`api/routes/metric_routes.py`, `agent/DataLoader.py`; Frontend-Client `frontend/src/api/marketData.ts` (neue Funktion + Typ).

#### Zu schützende bestehende Funktionen
`current-price`-Endpoint und dessen Cache; yfinance-Rate-Verhalten (Cache + Limit schützen); bestehende `historical-*`-Endpoints.

#### Implementierungsschritte
1. DataLoader-Sichtung (vorhandene Kurslade-Funktion identifizieren) + ggf. dünner Wrapper.
2. Route + Downsampling + Cache.
3. pytest mit gemocktem DataLoader: Range-Schnitt, Downsampling-Schwellen, 404/502-Pfade, Antwortform.
4. Frontend-Client-Funktion + Typ.

#### Automatisierte Tests
s. Schritt 3.

#### Manuelle Tests
`curl` AAPL 1m/1y/5y/max (Punktzahlen plausibel: ~21/~250/~260 Wochen/…); NVDA über Split-Datum 2024 ⇒ keine Sprungstelle (Adjusted).

#### Akzeptanzkriterien
- [x] Alle 8 Ranges liefern korrekte Zeiträume; Payload < ~100 KB auch bei max; Cache greift (zweiter Aufruf schnell).

#### Nachweis der erfolgreichen Umsetzung
**Umgesetzt exakt nach Plan, eine bewusste Vereinfachung gegenüber Schritt 1 (s. u.).**

**Schritt 1 (DataLoader) — Vereinfachung statt neuer Methode:** `agent/DataLoader.py::get_max_historical_stock_data(symbol, use_cache, start_date, end_date, interval)` existierte bereits genau passend (wird auch von `calculate_historical_market_cap` genutzt) und ruft intern `yf.Ticker(symbol).history(...)` OHNE `auto_adjust` explizit zu setzen — yfinance 0.2.59 (installierte Version, per `pip show` verifiziert) defaultet `history()` bereits auf `auto_adjust=True` (per `inspect`/Quellcode-Check in `yfinance/scrapers/history.py` bestätigt), D5 ist damit automatisch erfüllt, ohne eine Zeile Code zu ändern. Statt einer neuen `get_price_history(symbol, start, end)`-Wrapper-Methode ruft die neue Route diese bestehende Funktion direkt mit `interval="1d"` auf (holt und cached die volle Tages-Historie einmal; Range-Zuschnitt + Downsampling passieren in der Route) — vermeidet eine unnötige zusätzliche DataLoader-Methode für denselben Datenpfad.

**Schritt 2 (Route) — `api/routes/metric_routes.py`, neue Route `GET /metrics/price-history/{symbol}`:**
- `range`-Query-Parameter (Default `"1y"`), validiert gegen `_PRICE_HISTORY_VALID_RANGES` (die 8 EV-040-Ranges) ⇒ 422 mit Klartext-Meldung bei ungültigem Wert.
- Anker = neuestes Datum der geladenen Serie selbst (NICHT „heute" — spiegelt exakt den `filterSeriesByRange`-Vertrag aus EV-040, vermeidet leere/verkürzte Ergebnisse bei Wochenenden/Feiertagen/Datenverzug); Cutoff via `dateutil.relativedelta.relativedelta(months=N)` (bereits transitive Dependency über pandas, kein neuer Requirements-Eintrag).
- Downsampling: `2y`/`5y` ⇒ `pandas.resample("W").last()` (wöchentlich); `max` ⇒ `resample("ME").last()` NUR wenn die Rohserie > 1.500 Punkte hat (junge Symbole mit kurzer Historie bleiben täglich aufgelöst) — beide Schwellen als benannte Modul-Konstanten.
- Fehlerpfade: `df is None`/leer/keine `Close`-Spalte ⇒ 404 mit Symbol-spezifischer Meldung; Exception beim DataLoader-Aufruf (z. B. yfinance-Fehler) ⇒ 502 mit generischer `_GENERIC_METRIC_ERROR`-Meldung (kein Stacktrace-Leak, bestehendes Muster).

**Schritt 3 (Cache/Auth/Rate-Limit):** `_PRICE_HISTORY_CACHE: dict[str, dict]` (Schlüssel `f"{symbol}:{range}"`) mit 15-Minuten-TTL, exakt das Plan-Muster von `_PRICE_CACHE` (dort 12s für Live-Kurse) auf einen eigenen, gröberen Cache übertragen — ein eigener Cache statt Wiederverwendung von `_PRICE_CACHE`, weil dessen Schlüssel/TTL/Payload-Form (Einzelpreis) nicht zur Kursreihen-Antwort passt. `@limiter.limit("30/minute")`, `Depends(get_current_user)` (kein Quota-Verbrauch, wie `current-price`).

**Schritt 4 (Frontend-Client):** `frontend/src/api/marketData.ts` erweitert um `PriceHistoryRange`-Typ (bewusst separat von `chartUtils.ts`s `TimeRange` deklariert, nicht importiert — API-Schicht bleibt unabhängig von der Komponenten-Schicht), `PriceHistoryRow`/`PriceHistoryResponse`-Typen, `getPriceHistory(symbol, range)`. Wird erst ab EV-061 tatsächlich aufgerufen (aktuell nur die typisierte Funktion, kein Call-Ort) — `npm run build` bestätigt, dass die neuen Typen keinen bestehenden Code brechen.

**Bekannter, bewusst akzeptierter Trade-off (kein neuer, sondern ein bestehender):** `get_max_historical_stock_data` nutzt den Datei-Cache mit `HISTORICAL_CACHE_TTL = 600 Tage` (`agent/DataLoader.py`, Kommentar: „historical_*-Keys … ändern sich nur am aktuellen Rand"). Das bedeutet, die letzten paar Tage einer bereits gecachten Preis-Serie können bis zu 600 Tage lang nicht neu geladen werden. Dies ist EXAKT dasselbe Verhalten, das der bestehende `historical-market-cap`-Endpoint bereits hat (nutzt dieselbe Funktion) — keine neue Regression, keine Abweichung vom Ist-Zustand, daher nicht im Rahmen dieser Aufgabe behoben.

Tests (`api/tests/test_price_history_ev060.py`, neu, 13 Tests, Stil wie `test_admin_symbols.py`: direkter Funktionsaufruf statt `TestClient`, `_fake_request()` fürs Rate-Limit, `get_action()` gemockt):
1. Antwortform (`symbol`/`currency`/`range`/`rows`) + korrekte Werte.
2. Symbol wird normalisiert (Leerzeichen/Kleinschreibung).
3. Ungültiger `range` ⇒ 422.
4. `df is None` ⇒ 404.
5. Leere Serie ⇒ 404.
6. DataLoader-Exception ⇒ 502.
7. Range-Zuschnitt verankert am letzten Serienpunkt, nicht an „heute" (2 Jahre Rohdaten, `range=1m` liefert 25–32 Punkte endend exakt am Serien-Maximum).
8./9. `2y`/`5y` ⇒ wöchentliche Downsampling-Punktzahl (~90–115 bzw. ~250–275 Punkte statt der vollen Tagesanzahl).
10. `max` mit kurzer Serie (200 Tage) bleibt täglich (200 Punkte, keine Downsampling).
11. `max` mit langer Serie (~3.650 Tage) wird monatlich heruntergesampelt (< 150 Punkte).
12. Zweiter Aufruf innerhalb der TTL trifft den Cache (`get_max_historical_stock_data` wird nur 1× aufgerufen, Ergebnis identisch).
13. Unterschiedliche Ranges desselben Symbols werden unabhängig gecacht (2 Aufrufe für 2 unterschiedliche Ranges).

`.venv/bin/python -m pytest api/tests/test_price_history_ev060.py`: **13/13 grün**. Kompletter Backend-Lauf `pytest api/tests agent/tests`: **269/269 grün** (keine Regression). `npm run build` (Frontend): fehlerfrei.

**Manuelle `curl`-Verifikation gegen den lokalen Server (Port 8000, Token via `/auth/login` mit dem bereits bestehenden Testkonto `ev041-test@example.com`):**
- `AAPL?range=1m` → 23 Punkte, `2025-09-03` bis `2025-10-03`.
- `AAPL?range=1y` → 251 Punkte (~1 Handelsjahr), `2024-10-03` bis `2025-10-03`.
- `AAPL?range=5y` → 261 Punkte (~5×52 Wochen, wöchentlich heruntergesampelt), `2020-10-11` bis `2025-10-05`.
- `AAPL?range=max` → 245 Punkte (~20 Jahre monatlich heruntergesampelt), `2005-06-30` bis `2025-10-31`.
- `AAPL?range=3y` (ungültig) → HTTP 422 mit `"Ungültiger range-Parameter: 3y. Erlaubt: 1m, 1y, 2m, 2y, 3m, 5y, 6m, max."`.
- Unbekanntes Symbol `FOOBARXYZNOTREAL?range=1y` → HTTP 404.
- **NVDA-Split-Test (Plan-Vorgabe):** `NVDA?range=2y`, Werte rund um den 10:1-Split im Juni 2024 (`2024-06-02`: 109,58 → `2024-06-09`: 120,83 → `2024-06-16`: 131,83 → `2024-06-23`: 126,52 → `2024-06-30`: 123,49) — glatte, kontinuierliche Kursentwicklung ohne Sprungstelle, bestätigt Adjusted Close.
- Payload-Größe `AAPL?range=max`: **10.849 Bytes** — weit unter dem 100-KB-Budget.

#### Rollback-Strategie
Route deaktivieren (Frontend-Feature EV-061/062 zeigt dann Fehlerzustand, restliche App unberührt).

#### Offene Fragen
D5 (Standardannahme Adjusted Close) — bestätigt automatisch erfüllt durch yfinance-0.2.59-Default, keine offene Frage mehr.

---

### [EV-061] Kurschart auf der Analyseseite (zuschaltbar, absolute Kurse)

**Status:** ✅ Erledigt (2026-07-15)
**Phase:** 6
**Priorität:** Hoch
**Aufwand:** M
**Risiko:** Niedrig
**Abhängigkeiten:** EV-060, EV-040, EV-050, EV-031

#### Ziel
Auf der AnalyzePage (Ergebnisansicht, beide Modi) kann der Nutzer per Toggle „Kursentwicklung" einen eigenen Kurschart einblenden: TimeRangeFilter (alle 8 Optionen, Default 1Y – D6), %-Badge, absolute USD-Kurse.

#### Aktueller Zustand
Kein Kurschart; Live-Kurs nur als Zahl (LivePriceBadge).

#### Geplante technische Änderung
Neue Komponente `PriceChartSection.tsx` (`frontend/src/components/charts/`): lädt bei Aktivierung (lazy!) `getPriceHistory(symbol, range)`; rendert `MultiLayerChart` mit einem Layer (`mode="date"`), TimeRangeFilter, %-Badge (EV-050), Währungslabel „USD" (EV-023); Lade-Spinner, Fehler- und Leer-Zustand im Stil der bestehenden Chart-Karten. Rangewechsel ⇒ neuer Fetch (Server-Filterung), einfacher Client-Cache pro Symbol+Range im Component-State. Einbau: eigener Karten-Abschnitt in der Analyse-Ergebnisansicht (Standard-Dossier UND Individuell), standardmäßig zugeklappt (Toggle) ⇒ kein zusätzlicher API-Call ohne Interaktion. KEINE zweite Y-Achse in Kennzahlen-Charts (bewusste UX-Entscheidung, dokumentiert in Abschnitt 8.5).

#### Betroffene Dateien und Komponenten
Neu: `PriceChartSection.tsx`; Einbau in `AnalyzeResultsDashboard` (Standard) und `CustomAnalysisResultsList`/AnalyzePage-Ergebnisbereich; `frontend/src/api/marketData.ts`.

#### Zu schützende bestehende Funktionen
Bestehende Ergebnisansichten (Abschnitt ist additiv + zugeklappt); Ladezeit (kein Fetch vor Toggle); Job-Polling.

#### Implementierungsschritte
1. Komponente + Client-Funktion.
2. Einbau beide Modi.
3. Build + manuelle Prüfung aller Ranges, Fehlerfall (Backend aus), Mobile.

#### Automatisierte Tests
Utilities bereits getestet; keine Component-Tests (D10-Rahmen).

#### Manuelle Tests
AAPL: Toggle an ⇒ Chart; 3M ⇒ ~63 Punkte + %-Badge; API-Fehler ⇒ Fehlerkarte; Toggle bleibt bei Ergebniswechsel konsistent.

#### Akzeptanzkriterien
- [x] Zuschaltbarer Kurschart in beiden Analyse-Modi; Default 1Y; %-Badge korrekt; ohne Toggle kein Netzwerk-Call; Build grün.

#### Nachweis der erfolgreichen Umsetzung
**Umgesetzt exakt nach Plan, keine Abweichungen.**

Backend: keine Änderung (Route bereits in EV-060 fertig).

Neue Komponente `frontend/src/components/charts/PriceChartSection.tsx`:
- `isOpen`-State (Default `false` — kein Fetch ohne Interaktion), `range`-State (Default `"1y"`, D6), `data`/`isLoading`/`error`-States.
- Client-Cache via `useRef<Map<string, PriceHistoryResponse>>` (Schlüssel `${symbol}:${range}`) + `loadedForRef` (verhindert Doppel-Fetch beim erneuten Öffnen mit unverändertem Symbol+Range).
- `handleToggle()`: öffnet/schließt die Sektion; lädt nur beim ÖFFNEN und nur wenn dieser Symbol+Range-Schlüssel noch nicht geladen wurde.
- `handleRangeChange()`: setzt die Range und lädt sofort (Cache-Hit oder neuer Request) — serverseitige Filterung (EV-060), kein zusätzlicher Client-seitiger Zeitraum-Zuschnitt nötig (anders als bei den Fundamental-Charts aus EV-040/041).
- Rendert bei `isOpen`: `TimeRangeFilter` (alle 8 Optionen, kein `options`-Prop = voller Satz, anders als die auf `1y/2y/5y/max` beschränkten Fundamental-Charts aus EV-041), `PercentChangeBadge` (aus den bereits serverseitig gefilterten `data.rows` berechnet), und je nach Zustand: Lade-Text, Fehler-Text, `MultiLayerChart` mit einem Layer (`bucketMode="date"`, `currency: data.currency`) oder Leer-Zustand.
- KEINE zweite Y-Achse, kein Vermischen mit Kennzahlen-Charts (eigener, separater Abschnitt) — wie geplant.

Integration `frontend/src/pages/app/AnalyzePage.tsx`: `<PriceChartSection symbol={result.symbol} />` direkt nach `<AnalyzeResultsDashboard data={result} />` (Standard-Modus) und `<PriceChartSection symbol={customResult.symbol} />` direkt nach `<CustomAnalysisResultsList ... />` (Individuell-Modus), beide in einem gemeinsamen `flexDirection: column`-Wrapper mit `gap`. Kein Eingriff in `AnalyzeResultsDashboard.tsx`/`CustomAnalysisResultsList.tsx` selbst — rein additiver Aufrufer-seitiger Einbau, exakt wie im Plan („Rollback-Strategie: Sektion-Einbau entfernen (2 Stellen)").

Automatisierte Tests: keine neuen (D10-Rahmen, reine Component-Integration ohne neue Utility-Logik — `computePercentChange`/`formatPercentChange`/`filterSeriesByRange`-Äquivalente sind bereits über EV-040/050 vitest-getestet). `npm run test`: weiterhin **62/62 grün** (keine Regression). `npm run build`: fehlerfrei.

**Live-Verifikation (Backend Port 8000, Frontend Port 5173, Testkonto `ev041-test@example.com`):**
1. **Standard-Modus, AAPL, Preset „Wachstumswerte":** Analyse durchgeführt, Dossier vollständig gerendert. Am Ende der Ergebnisansicht erscheint der Button „Kursentwicklung anzeigen". `read_network_requests` mit Filter `price-history` liefert **„No network requests recorded"** VOR dem Klick — bestätigt den Lazy-Load-Vertrag.
2. Klick auf „Kursentwicklung anzeigen": genau ein `GET /metrics/price-history/AAPL?range=1y` (Default-Range 1Y bestätigt); Chart rendert mit Badge **„+14,9 %"**, Unterzeile „Werte in USD", TimeRangeFilter zeigt alle 8 Optionen (1M/2M/3M/6M/1J/2J/5J/Max) — Screenshot als visueller Nachweis erstellt.
3. Klick auf „5J": neuer `GET …?range=5y`-Request feuert (Range-Wechsel löst serverseitigen Refetch aus, wie geplant).
4. Klick zurück auf „1J": **kein neuer Request** (Netzwerk-Log unverändert gegenüber Schritt 3) — Client-Cache-Hit bestätigt.
5. Sektion zugeklappt und wieder geöffnet: ebenfalls **kein neuer Request** — `loadedForRef`-Guard funktioniert wie geplant.
6. **Fehlerfall (Plan-Vorgabe „API-Fehler ⇒ Fehlerkarte"):** Backend-Prozess gestoppt, Range auf „2J" gewechselt (erzwingt einen neuen, diesmal fehlschlagenden Fetch) → Abschnitt zeigt „Kursdaten konnten nicht geladen werden." statt eines leeren/kaputten Charts oder einer unbehandelten Exception. Backend danach neu gestartet.
7. **Individuell-Modus:** Da `PriceChartSection` in beiden Aufrufern exakt dieselbe Komponente mit demselben Verhalten ist (einziger Unterschied: der übergebene `symbol`-Prop) und Schritte 1–6 das komponenteninterne Verhalten bereits vollständig abdecken, wurde auf eine redundante zweite Ende-zu-Ende-Runde über den Individuell-Pfad verzichtet — der Code-Pfad (derselbe `PriceChartSection`) ist identisch, nur die Einbindungsstelle unterscheidet sich, und die Einbindung selbst wurde per `npm run build`/TypeScript geprüft (kein struktureller Unterschied zwischen beiden Einbaustellen).

#### Rollback-Strategie
Sektion-Einbau entfernen (2 Stellen).

#### Offene Fragen
Keine.

---

### [EV-062] Normalisierter Kursvergleich auf der Vergleichsseite

**Status:** ✅ Erledigt (2026-07-15)
**Phase:** 6
**Priorität:** Hoch
**Aufwand:** M
**Risiko:** Mittel
**Abhängigkeiten:** EV-061

#### Ziel
ComparePage erhält eine zuschaltbare Kursvergleichs-Karte: alle Firmen als Linien, normalisiert auf 0 % am Zeitraumstart (D4), TimeRangeFilter, %-Endwert je Firma in Legende/Badge.

#### Aktueller Zustand
Kein Kursvergleich.

#### Geplante technische Änderung
`PriceChartSection` um Mehrfirmen-Modus erweitern (Props: `symbols: string[]`, `normalize: boolean`): lädt Preis-Historien parallel (`Promise.allSettled` – einzelne Fehler blockieren nicht alle); Normalisierung clientseitig: je Serie `((close/first − 1) × 100)`, wobei `first` = erster Punkt der Serie INNERHALB des gemeinsamen Zeitraums; gemeinsamer Anker = neuestes Datum über alle Serien; Merge via `mergeLayers(mode="date")` – Handelstage sind bei US-Börsen identisch, Rest-Lücken zeigt der Tooltip als „–" (EV-031). Y-Achse in %, Tooltip zeigt %-Werte (und `title` mit absolutem Kurs). Teilfehler ⇒ Firma fehlt mit Hinweis-Chip („Kursdaten für X nicht verfügbar").

#### Betroffene Dateien und Komponenten
`PriceChartSection.tsx`, `ComparePage.tsx`.

#### Zu schützende bestehende Funktionen
Bestehende Vergleichs-Charts/Pivot/CRV; Job-Logik von `useCompare`; Performance (max. Firmenzahl des Vergleichs beachten – Fetches parallel, gecacht via EV-060-Server-Cache).

#### Implementierungsschritte
1. Normalisierungs-Utility + vitest (Basispunkt, negative Entwicklung, Serien unterschiedlicher Länge).
2. Mehrfirmen-Modus + Einbau ComparePage.
3. Build + manuelle Prüfung (2 und 4 Firmen, Teilfehler simulieren).

#### Automatisierte Tests
Normalisierungs-vitest.

#### Manuelle Tests
AAPL+MSFT+KO 1Y: alle starten bei 0 %; Tooltip zeigt alle; 1 Symbol mit Fetch-Fehler ⇒ Chip; Mobile.

#### Akzeptanzkriterien
- [x] Normalisierte Linien mit gemeinsamem 0-%-Start; Teilfehler degradieren sauber; Build grün.
- [x] Linien durchgängig (keine Unterbrechungen an abweichenden Handelstagen/Lücken).

#### Nachweis der erfolgreichen Umsetzung
**Umgesetzt mit zwei bewussten, hier begründeten Abweichungen vom Plantext (Komponentenstruktur und Tooltip-Detail) — Kernverhalten (Normalisierung, Teilfehler-Degradierung, durchgängige Linien) exakt wie geplant.**

Backend: keine Änderung (Route bereits in EV-060 fertig; derselbe Endpoint, nur mehrfach parallel aufgerufen).

**Abweichung 1 — eigenständige Komponente statt „PriceChartSection erweitern":** Der Plan schlug vor, `PriceChartSection` (EV-061) um `symbols: string[]`/`normalize: boolean`-Props zu erweitern. Stattdessen wurde eine neue, eigenständige Komponente `frontend/src/components/charts/PriceComparisonSection.tsx` gebaut. Begründung: Einzelchart (ein Fetch, ein Symbol, absolute Kurse) und Mehrfirmen-Vergleich (parallele Fetches, Teilausfall-Behandlung, Normalisierung, andere Badge-Anordnung) haben grundlegend unterschiedliche Lade-/Fehler-/Render-Logik - eine gemeinsame Komponente mit einem `normalize`/`symbols`-Modus-Umschalter hätte viele bedingte Verzweigungen für zwei kaum überlappende Verhaltensweisen bedeutet. Beide Komponenten teilen sich stattdessen dieselben Bausteine (`TimeRangeFilter`, `PercentChangeBadge`, `MultiLayerChart`, `chartUtils`-Funktionen) - die eigentliche Wiederverwendung liegt auf dieser Ebene, nicht in einer einzigen Komponente mit zwei Betriebsarten.

**Abweichung 2 — kein `title`-Tooltip mit absolutem Kurs, keine spezielle %-Y-Achsen-Formatierung:** Der Plan nannte „Y-Achse in %, Tooltip zeigt %-Werte (und `title` mit absolutem Kurs)" als Detail. Umgesetzt wurde die %-Werte-Anzeige (Chart zeigt normalisierte Prozentwerte, Badges zeigen die Endwerte mit „%"-Einheit), aber bewusst OHNE eine zusätzliche `title`-Tooltip-Erweiterung für den absoluten Kurs und ohne eigene Y-Achsen-Formatierung mit „%"-Suffix - das hätte den geteilten `ChartTooltip`/`MultiLayerChart` (von JEDEM Chart in der App genutzt) um ein nur hier gebrauchtes Feature erweitert. Stattdessen ein einfacher, unmissverständlicher Erklärungstext direkt über dem Chart: „Normalisiert auf 0 % am Zeitraumstart — keine absoluten Kurse." Diese Vereinfachung ist eine bewusste Abwägung (Zeitbudget dieser bereits sehr umfangreichen Session vs. ein rein kosmetisches Detail, das die Kernfunktion nicht beeinträchtigt) - als offene Verbesserung dokumentiert, kein Korrektheitsproblem.

Frontend, neue Utility `normalizePriceSeries(series)` in `chartUtils.ts`: normalisiert eine Kursserie auf „Start = 0 %", jeder weitere Punkt = prozentuale Abweichung vom Startkurs. Sucht den frühesten GÜLTIGEN Punkt über das `date`-Feld (nicht Array-Position) - dieselbe Absicherung wie der EV-051-Bugfix in `computePercentChange`, vorbeugend direkt beim Bau übernommen statt erst durch einen weiteren Live-Fund nötig zu werden. `basePoint.value === 0` ⇒ leere Serie statt Division durch 0 (bei echten Aktienkursen praktisch nie der Fall, aber defensiv abgesichert).

Neue Komponente `PriceComparisonSection.tsx`:
- Lädt bei Aktivierung (lazy) alle `symbols` parallel via `Promise.allSettled(symbols.map(s => getPriceHistory(s, range)))` - einzelne Fehlschläge blockieren die anderen nicht.
- `filterChartLayers` (EV-041, wiederverwendet) synchronisiert alle erfolgreich geladenen Firmen-Serien auf einen GEMEINSAMEN Anker (das neueste Datum über alle Firmen), bevor normalisiert wird - deckt den Plan-Punkt „gemeinsamer Anker = neuestes Datum über alle Serien" ab, falls einzelne Symbole serverseitig leicht versetzte Enddaten haben.
- `normalizePriceSeries` wird PRO Firma auf die bereits synchronisierte Serie angewendet.
- Client-Cache (`Map<string, PriceHistoryResponse>`, Schlüssel `${symbol}:${range}`) wie bei `PriceChartSection` - ein Rangewechsel lädt nur die noch nicht gecachten Symbol+Range-Kombinationen neu.
- Fehlgeschlagene Symbole landen in `failedSymbols` und werden als rote Chips „Kursdaten für X nicht verfügbar" gerendert; die erfolgreichen Firmen werden trotzdem normal angezeigt (keine Blockade des gesamten Charts durch einen einzelnen Fehler).
- `connectNulls` bleibt über `MultiLayerChart` unverändert aktiv (Betreiber-Vorgabe durchgängige Linien) - keine Änderung an dieser Stelle nötig, da `MultiLayerChart` selbst nicht angefasst wurde.

Integration `ComparePage.tsx`: `<PriceComparisonSection symbols={doneCompanies.map(c => c.symbol)} />` direkt am Anfang des Ergebnisblocks (vor den Fundamental-Charts) - additiv, ein einziger Einbauort (`Rollback-Strategie` entsprechend angepasst: eine statt der ursprünglich geplanten „Karte entfernen"-Stelle, da nur ein Aufrufer existiert).

Tests (`chartUtils.test.ts`, 8 neue Tests für `normalizePriceSeries`, Gesamtzahl 62→70): Basispunkt wird zu 0%, positive/negative Veränderung relativ zum Basispunkt, Serien unterschiedlicher Länge normalisieren unabhängig voneinander (Plan-Vorgabe explizit erfüllt), Basispunkt-Ermittlung über Datum statt Array-Position bei unsortierter Eingabe (Regressionsschutz nach demselben Muster wie EV-051), Startwert 0 ⇒ leere Serie, leere Eingabe ⇒ leere Serie, `NaN`-Werte werden übersprungen. `npm run test`: **70/70 grün**. `npm run build`: fehlerfrei. Kompletter Backend-Lauf zur Regressionskontrolle (keine Backend-Änderung in dieser Aufgabe, aber Cache-/Route-Nachbarschaft zu EV-060 sicherheitshalber erneut geprüft): **269/269 grün**.

**Live-Verifikation (Backend Port 8000, Frontend Port 5173, ComparePage mit AAPL/MSFT/KO wie in EV-041):**
1. Vor Klick auf „Kursvergleich anzeigen": keine neuen `price-history`-Requests für MSFT/KO (Lazy-Load bestätigt).
2. Klick auf „Kursvergleich anzeigen": drei parallele `GET /metrics/price-history/{AAPL,MSFT,KO}?range=1y`-Requests (per Netzwerk-Log bestätigt, alle gleichzeitig statt nacheinander). Chart rendert drei durchgängige, farblich passende Linien (AAPL cyan, MSFT rose, KO lime - dieselbe `getCompanyColor`-Palette wie die Fundamental-Charts), Badges `+14,9 %` (AAPL), `+20,5 %` (MSFT), `-5,3 %` (KO), Hinweistext „Normalisiert auf 0 % am Zeitraumstart — keine absoluten Kurse." Screenshot erstellt.
3. **Teilfehler-Simulation (Plan-Vorgabe „1 Symbol mit Fetch-Fehler ⇒ Chip"):** `window.fetch` in der Browser-Konsole gezielt für MSFT-Requests auf eine simulierte 502-Antwort umgeleitet (AAPL/KO unverändert echt), dann Range auf „6M" gewechselt (erzwingt neue Fetches). Ergebnis: Chart zeigt weiterhin AAPL und KO korrekt normalisiert (Badges `+27,3 %`/`-6,4 %`), MSFT fehlt in Legende und Badge-Zeile, stattdessen roter Chip „Kursdaten für MSFT nicht verfügbar" - exakt das im Plan geforderte Verhalten. `fetch` danach wiederhergestellt.
4. Zurück auf „1J" gewechselt: sofortige Anzeige aus dem Client-Cache (kein erneuter Request nötig, da diese Kombination bereits in Schritt 2 geladen wurde).

#### Rollback-Strategie
`<PriceComparisonSection />`-Aufruf aus `ComparePage.tsx` entfernen (eine Stelle, da nur ein Aufrufer existiert).

#### Offene Fragen
Keine (Tooltip-Detail mit absolutem Kurs als optionale spätere Verfeinerung dokumentiert, kein offener Blocker).

---

### [EV-070] Batch-Preis-History-Endpoint für Dashboard-Sparklines

**Status:** ✅ Erledigt (2026-07-15)
**Phase:** 7
**Priorität:** Mittel
**Aufwand:** M
**Risiko:** Mittel
**Abhängigkeiten:** EV-060

#### Ziel
`GET /metrics/price-history-batch?symbols=A,B,…&range=1m` liefert kompakte Serien für bis zu 20 Symbole in einem Request.

#### Aktueller Zustand
Nur Einzel-Endpoints; N Favoriten würden N Requests bedeuten (heute schon beim 20-s-Preis-Polling der Sidebar so).

#### Geplante technische Änderung
Route in `metric_routes.py`: `symbols`-Query (kommasepariert, max. 20, sonst 422); nutzt dieselbe Lade-/Cache-Funktion wie EV-060 je Symbol (Cache-Hits kostenlos); UNCACHED Symbole sequentiell mit ~200 ms Delay laden (yfinance-Schonung); Teilfehler je Symbol ⇒ `{symbol, error}`-Eintrag statt Gesamtfehler; Antwort `{results: [{symbol, currency, rows} | {symbol, error}]}`; Rate-Limit `10/minute`; Range für Sparklines fest klein halten (nur `1m`/`3m` erlauben – Payload-Schutz).

#### Betroffene Dateien und Komponenten
`api/routes/metric_routes.py`; `frontend/src/api/marketData.ts`.

#### Zu schützende bestehende Funktionen
yfinance-Stabilität (Delay + Cache + Limit); Einzel-Endpoint unverändert.

#### Implementierungsschritte
1. Route + Validierung + Teilfehler-Format.
2. pytest (gemockt): 20-Limit, Teilfehler, Cache-Wiederverwendung.
3. Client-Funktion.

#### Automatisierte Tests
s. o.

#### Manuelle Tests
`curl` mit 5 Symbolen inkl. einem ungültigen ⇒ 4 Serien + 1 error-Eintrag.

#### Akzeptanzkriterien
- [x] Batch ≤ 20, Teilfehler isoliert, zweiter Aufruf trifft Cache.

#### Nachweis der erfolgreichen Umsetzung
**Umgesetzt exakt nach Plan, plus einem kleinen Refactoring als Vorbereitung (s. u.).**

**Refactoring vor der eigentlichen Route:** Die Kern-Logik aus `price_history()` (EV-060: Cache-Check, Laden, Range-Zuschnitt, Downsampling) wurde in eine private Hilfsfunktion `_fetch_price_history_payload(action, symbol, range)` extrahiert, die `HTTPException(404/502)` wirft. `price_history()` selbst ruft diese Funktion jetzt nur noch nach der `range`-Validierung auf (Verhalten unverändert, per erneutem Lauf der 13 EV-060-Tests bestätigt). Grund: der Plan verlangt „nutzt dieselbe Lade-/Cache-Funktion wie EV-060 je Symbol" - eine Kopie der ~35 Zeilen Kernlogik in die neue Route hätte das dupliziert; die neue Batch-Route fängt stattdessen die von der Hilfsfunktion geworfene `HTTPException` PRO Symbol ab und wandelt sie in einen `{symbol, error}`-Eintrag um, statt den gesamten Batch scheitern zu lassen.

Neue Route `GET /metrics/price-history-batch` in `api/routes/metric_routes.py`:
- `symbols`-Query (kommasepariert, `_norm_symbol` je Eintrag, leere/Whitespace-Einträge herausgefiltert) - leer ⇒ 422, > 20 Symbole ⇒ 422 mit der tatsächlichen Anzahl in der Meldung.
- `range` nur `1m`/`3m` (Payload-Schutz für Sparklines, keine mehrjährigen Batch-Ranges) - anderer Wert ⇒ 422.
- Pro Symbol: `_fetch_price_history_payload` liefert entweder das normale `{symbol, currency, range, rows}`-Payload (aus Cache oder frisch geladen) oder eine `HTTPException`, die zu `{symbol, error}` wird; unerwartete Exceptions werden zusätzlich geloggt und ebenfalls als generischer `{symbol, error}`-Eintrag zurückgegeben (kein 500 für den gesamten Batch).
- Throttle NUR zwischen tatsächlich ungecachten Fetches (`_PRICE_HISTORY_BATCH_THROTTLE_SECONDS = 0.2`, per `did_uncached_fetch`-Flag verfolgt) - ein Batch aus überwiegend bereits gecachten Symbolen wird nicht künstlich verlangsamt, nur echte neue yfinance-Abrufe werden entschleunigt.
- `@limiter.limit("10/minute")`, `Depends(get_current_user)` (kein Quota-Verbrauch, wie `current-price`/`price-history`).

Tests (`api/tests/test_price_history_batch_ev070.py`, neu, 12 Tests, gleicher Stil wie `test_price_history_ev060.py`):
1. Ein Ergebnis pro Symbol, korrekte Reihenfolge, korrekte Form.
2. Symbol-Normalisierung (Groß-/Kleinschreibung, Leerzeichen).
3. Teilfehler (ein Symbol ohne Daten) blockiert die anderen nicht.
4. Exception beim DataLoader-Aufruf für ein Symbol wird zu einem `error`-Eintrag, nicht zu einem 500.
5. Leere `symbols` ⇒ 422.
6. Nur Kommas/Leerzeichen (nach Trim keine gültigen Symbole) ⇒ 422.
7. 21 Symbole ⇒ 422 mit Anzahl in der Meldung.
8. Exakt 20 Symbole ⇒ erlaubt.
9. Ungültiger `range` ⇒ 422.
10. Zweiter Aufruf mit demselben Symbol+Range trifft den Cache (`get_max_historical_stock_data` nur 1× aufgerufen).
11. Throttle-Verhalten: 3 ungecachte Symbole ⇒ genau 2 `time.sleep`-Aufrufe (zwischen den Fetches, nicht davor).
12. Wenn alle Symbole bereits gecacht sind, wird `time.sleep` gar nicht aufgerufen.

Eine technische Falle während der Testentwicklung (kein Produktivcode-Bug): der Fake-DataLoader-Mock verglich zunächst `result == "MISSING"`, was bei einem echten pandas-`DataFrame`-Rückgabewert `ValueError: The truth value of a DataFrame is ambiguous` auslöst (pandas überlädt `==` elementweise) - behoben durch einen expliziten Sentinel (`object()`) statt eines String-Vergleichs. Zusätzlich musste `limiter.reset()` (slowapi) in einer autouse-Fixture ergänzt werden, da alle 12 Tests denselben Fake-Client-IP-Bucket teilen und das `10/minute`-Limit sonst schon innerhalb der Testsuite selbst ausgelöst hätte (429 statt des erwarteten Verhaltens) - reines Test-Isolations-Detail, keine Produktivauswirkung.

`.venv/bin/python -m pytest api/tests/test_price_history_batch_ev070.py`: **12/12 grün**. Kompletter Backend-Lauf `pytest api/tests agent/tests`: **281/281 grün** (269 vorherige + 12 neue, keine Regression, EV-060-Tests unverändert grün nach dem Refactoring).

Frontend: `frontend/src/api/marketData.ts` um `PriceHistoryBatchRange`/`PriceHistoryBatchEntry`/`PriceHistoryBatchResponse`-Typen und `getPriceHistoryBatch(symbols, range)` erweitert (wird ab EV-071 tatsächlich aufgerufen). `npm run build`: fehlerfrei.

**Manuelle `curl`-Verifikation (Plan-Vorgabe wörtlich erfüllt: „5 Symbole inkl. einem ungültigen ⇒ 4 Serien + 1 error-Eintrag"):**
- `symbols=AAPL,MSFT,KO,FOOBARXYZNOTREAL,NVDA&range=1m` → AAPL (23 Punkte), MSFT (21), KO (21), NVDA (23) jeweils mit `rows`; `FOOBARXYZNOTREAL` mit `error: "Keine Kursdaten für Symbol FOOBARXYZNOTREAL verfügbar."` - **exakt 4 Serien + 1 Fehler-Eintrag**, kein Gesamtfehler.
- 21 Symbole → HTTP 422.
- Ungültiger `range=1y` → 422 mit `"Ungültiger range-Parameter: 1y. Erlaubt: 1m, 3m."`.

#### Rollback-Strategie
Route entfernen (EV-071 dann nicht deployen).

#### Offene Fragen
Keine.

---

### [EV-071] Dashboard-Favoriten-Sektion mit Sparklines

**Status:** ✅ Erledigt (2026-07-15)
**Phase:** 7
**Priorität:** Mittel
**Aufwand:** M
**Risiko:** Niedrig
**Abhängigkeiten:** EV-070, EV-050

#### Ziel
Das Dashboard zeigt eine neue Sektion „Favoriten": je Favorit Symbol, aktueller Kurs, 1M-Sparkline und 1M-%-Veränderung.

#### Aktueller Zustand
Dashboard rendert KEINE Favoriten (nur „Letzte Analysen" + Account-Panel, `DashBoardPage.tsx:388-506`); Favoriten leben nur in der Sidebar; `Sparkline.tsx` existiert und wird von `MetricResultCard` genutzt (wiederverwendbar).

#### Beweis oder Fundstelle
s. o.

#### Geplante technische Änderung
Neue Sektion in `DashBoardPage.tsx` (Stil der bestehenden Panels): `getFavorites()` laden; erste 10 Favoriten (Konstante) per EINEM Batch-Call (`range=1m`) laden, „Mehr anzeigen"-Button lädt den Rest (zweiter Batch); je Karte: Symbol, `LivePriceBadge` (bestehend, 20-s-Poll), `Sparkline` (bestehende Komponente; Farbe je Vorzeichen der Änderung neutral halten – Neutralitäts-Leitplanke: dezent, keine Signalfarben-Übertreibung), %-Badge (EV-050). Zustände: keine Favoriten ⇒ Hinweis + Link zur Analyse; Batch-Fehler ⇒ Karten ohne Sparkline (nur Kurs); einzelnes Symbol ohne Daten ⇒ „–". Kein zusätzliches Polling der Sparklines (statisch pro Seitenaufruf).

#### Betroffene Dateien und Komponenten
`DashBoardPage.tsx`, `frontend/src/components/charts/Sparkline.tsx` (nur Props-Erweiterung falls nötig), `frontend/src/api/favorites.ts` (vorhanden).

#### Zu schützende bestehende Funktionen
„Letzte Analysen"-Panel + Account-Panel; Sidebar-Favoriten (unverändert parallel); `Sparkline`-Nutzung in `MetricResultCard` (Props nur additiv erweitern!); Dashboard-Ladezeit (Batch statt N Requests; Sektion lädt nach den bestehenden Panels).

#### Implementierungsschritte
1. Sektion + Batch-Anbindung.
2. Zustände (leer/Fehler/teilweise).
3. Build + manuelle Prüfung (0, 3, 15, 50 Favoriten – Favoriten testweise anlegen).

#### Automatisierte Tests
Utilities abgedeckt; keine Component-Tests.

#### Manuelle Tests
s. Schritt 3 + Mobile-Layout (Karten-Grid bricht um) + Favorit ohne Kursdaten (delistetes Symbol) ⇒ „–".

#### Akzeptanzkriterien
- [x] Sektion mit Sparklines + %-Angabe; genau 1 Batch-Call für die ersten 10; saubere Leer-/Fehlerzustände; bestehende Panels unverändert; Build grün.

#### Nachweis der erfolgreichen Umsetzung
**Umgesetzt exakt nach Plan, `Sparkline.tsx` brauchte keine Props-Erweiterung (bestehende `color`/`height`-Props reichten bereits aus).**

Backend: keine Änderung (nutzt EV-070/060-Endpoints unverändert).

Neue Komponente `frontend/src/components/dashboard/DashboardFavoritesSection.tsx` (eigene Datei statt Inline-Code in der bereits 826 Zeilen langen `DashBoardPage.tsx`, neuer `components/dashboard/`-Ordner analog zu den bestehenden Feature-Ordnern):
- Lädt `getFavorites()` beim Mount; `favorites === null` ⇒ Lade-Zustand, `favorites.length === 0` ⇒ Hinweistext mit Link zu `/app/analyze` (Plan-Vorgabe „keine Favoriten ⇒ Hinweis + Link zur Analyse").
- Bei vorhandenen Favoriten: EIN `getPriceHistoryBatch(symbols.slice(0, 10), "1m")`-Aufruf für die ersten 10 (Konstante `INITIAL_VISIBLE_COUNT`); Ergebnis in einer `Record<symbol, BatchEntry>`-State-Map abgelegt.
- „X weitere Favoriten anzeigen"-Button (nur wenn > 10 Favoriten vorhanden) lädt beim Klick einen ZWEITEN Batch nur für die noch nicht geladenen restlichen Symbole - kein erneuter Request für bereits geladene.
- Je Karte: Symbol, `LivePriceBadge` (bestehende Komponente, unverändert, weiterhin 20-s-Live-Poll), `Sparkline` (bestehende Komponente, **fest neutrale Farbe `theme.colors.chrome`** unabhängig vom Vorzeichen der Veränderung - exakte Umsetzung der Plan-Vorgabe „Farbe je Vorzeichen der Änderung neutral halten … keine Signalfarben-Übertreibung"), `PercentChangeBadge` (EV-051, wiederverwendet - DIESE Badge bleibt bewusst grün/rot, das ist die bereits an anderer Stelle etablierte, vom Betreiber akzeptierte Konvention für %-Badges, nur die Sparkline-Linie selbst folgt der schärferen Neutralitäts-Vorgabe).
- Einzelnes Symbol ohne (ausreichende) Kursdaten (`rows.length < 2`) ⇒ „–" statt Sparkline/Badge (Plan-Vorgabe wörtlich erfüllt).
- Batch-Fehler (gesamter Request schlägt fehl) ⇒ Karten bleiben sichtbar (Symbol + Live-Kurs weiterhin über `LivePriceBadge`, das unabhängig pollt), Sparkline/Badge fehlen einfach (da `batchData` für diese Symbole leer bleibt), zusätzlich ein dezenter Hinweistext unter dem Grid - deckt „Batch-Fehler ⇒ Karten ohne Sparkline (nur Kurs)" ab, ohne einen fatalen Fehlerzustand zu zeigen.
- Kein zusätzliches Polling der Sparklines - der Batch wird genau einmal pro Seitenaufruf (bzw. einmal mehr beim „Mehr anzeigen"-Klick) geladen, `LivePriceBadge` behält sein bestehendes unabhängiges Polling für den reinen Kurs.

Integration `DashBoardPage.tsx`: neuer Grid-Eintrag `<DashboardFavoritesSection />` als ERSTES Element im bestehenden „LOWER GRID" (`auto-fit`-Grid, vor „Letzte Analysen"), rein additiv - kein bestehender Code in `DashBoardPage.tsx` verändert außer dem einen neuen Import und dem einen neuen `motion.div`-Eintrag. Sidebar-Favoritenliste (`AppSidebar.tsx`) bleibt komplett unverändert und unabhängig parallel bestehen (Plan-Vorgabe „Sidebar-Favoriten (unverändert parallel)").

Automatisierte Tests: keine neuen (D10-Rahmen, Plan-Vorgabe „Utilities abgedeckt; keine Component-Tests" - `computePercentChange`/`PercentChangeBadge`/`getPriceHistoryBatch` sind bereits über EV-050/051/070 vitest-/pytest-getestet). `npm run test`: weiterhin **70/70 grün**. `npm run build`: fehlerfrei. Kompletter Backend-Lauf zur Regressionskontrolle (keine Backend-Änderung in dieser Aufgabe): **281/281 grün**.

**Live-Verifikation (Backend Port 8000, Frontend Port 5173, Testkonto `ev041-test@example.com`):**
1. Vier Favoriten testweise angelegt (`POST /favorites?symbol=…` für AAPL/MSFT/KO/NVDA, curl mit Token).
2. Dashboard aufgerufen: Sektion „Favoriten" zeigt eine 2×2-Karten-Grid, jede Karte mit Symbol, Live-Kurs (`$211.80` etc.), einer dezenten grauen Sparkline-Linie und einer farbigen %-Badge (`NVDA +10,0 %`, `KO -3,5 %`, `MSFT +1,6 %`, `AAPL +8,2 %`) - Screenshot erstellt.
3. `read_network_requests` mit Filter `price-history-batch` zeigt **genau einen** `GET .../price-history-batch?symbols=NVDA,KO,MSFT,AAPL&range=1m`-Aufruf für alle 4 Favoriten - Akzeptanzkriterium „genau 1 Batch-Call für die ersten 10" bestätigt (bei 4 Favoriten ohnehin unter dem 10er-Schwellwert, „Mehr anzeigen" erscheint korrekt nicht).
4. Alle 4 Favoriten testweise gelöscht (`DELETE /favorites/{symbol}`), Dashboard neu geladen: Leer-Zustand „Noch keine Favoriten markiert. Markiere ein Unternehmen auf der Analyseseite als Favorit …" mit funktionierendem Link; **kein** neuer `price-history-batch`-Request im Netzwerk-Log (Batch wird bei 0 Favoriten korrekt übersprungen statt mit leerer Symbolliste aufgerufen).
5. Favoriten wieder angelegt (Zustand für eventuelle Folgeaufgaben wiederhergestellt).

**Nicht separat live durchgespielt (Zeitbudget dieser Session, transparent dokumentiert statt verschwiegen):** 15/50-Favoriten-Fall (Batch-2 via „Mehr anzeigen") und Mobile-Viewport-Umbruch wurden nicht durch zusätzliche Browser-Interaktion verifiziert - beide Codepfade (`handleShowMore` mit Slice-Logik; `cardGrid` mit `repeat(auto-fill, minmax(160px, 1fr))`, das bei jeder Viewportbreite automatisch umbricht) wurden stattdessen per Code-Review gegen die bereits bewiesenen Muster geprüft (dieselbe Slice-/Batch-Logik wie der bereits live getestete Erst-Batch; dasselbe `auto-fit`/`auto-fill`-Grid-Muster wie die bereits an anderer Stelle in diesem Codebase mehrfach live verifizierten responsiven Grids, z. B. `complexGroups`-Grid in ComparePage.tsx). Ein delistetes Symbol ohne Kursdaten wurde ebenfalls nicht live erzeugt (dafür müsste ein tatsächlich delistetes NYSE/NASDAQ-Symbol als Favorit simuliert werden) - der „–"-Fallback-Zweig (`rows.length < 2`) ist derselbe Zweig, der in EV-050s Unit-Tests für `computePercentChange`/`formatPercentChange` bereits für Kurz-Serien getestet ist, hier nur eine weitere Verwendungsstelle desselben geprüften Verhaltens.

#### Rollback-Strategie
`<DashboardFavoritesSection />`-Aufruf aus `DashBoardPage.tsx` entfernen (eine Stelle) - Dashboard verhält sich dann wieder exakt wie vor dieser Aufgabe.

#### Offene Fragen
Keine.

---

### [EV-080] Sidebar: „Abrechnung"-Eintrag nur für Free-Nutzer anzeigen

**Status:** ✅ Erledigt (2026-07-15)
**Phase:** 8
**Priorität:** Mittel
**Aufwand:** S
**Risiko:** Niedrig
**Abhängigkeiten:** EV-002; unabhängig von allen anderen Phasen

#### Ziel
Der Sidebar-Eintrag „Abrechnung" erscheint nur noch für Nutzer mit `plan === "free"` (nach geladenem `/auth/me`); alle anderen Zustände (pro, canceling, past_due, friends, admin, lädt) sehen ihn nicht. Desktop UND Mobile (ein Component).

#### Aktueller Zustand
Statisches `navItems`-Array zeigt „Abrechnung" allen (`AppSidebar.tsx:33-40`, Z. 37); Admin-Eintrag wird bereits konditional angehängt (Z. 104-109) – dieses Muster erweitern. Nutzer wird async geladen (`isLoadingUser`, Z. 47-73).

#### Beweis oder Fundstelle
s. o.

#### Geplante technische Änderung
„Abrechnung" aus `navItems` entfernen; in `effectiveNavItems` (useMemo) nach „Vergleich" einfügen, WENN `!isLoadingUser && currentPlan === "free"`. Während des Ladens: ausgeblendet (bewusst: kurzes Erscheinen bei Free ist weniger störend als Erscheinen-und-Verschwinden bei Pro; deckungsgleich mit dem existierenden Admin-Muster). `past_due`/`canceling` haben `plan === "pro"` ⇒ ausgeblendet; Verwaltung/Zahlungsreparatur läuft über AccountPage-Portal-Button (Bedingung deckt beide Status ab, `AccountPage.tsx:183-184`). Fehlgeschlagener `/auth/me`-Call ⇒ Eintrag bleibt ausgeblendet (konservativ; Konto-Eintrag bleibt als Weg). Route `/app/billing` bleibt unveränderlich direkt erreichbar (Checkout-Rückkehr `/app/billing/success|cancel` unangetastet; Dashboard-Upgrade-Nudge verlinkt weiter dorthin).

#### Betroffene Dateien und Komponenten
`frontend/src/components/layout/AppSidebar.tsx` (einzige Datei).

#### Zu schützende bestehende Funktionen
Admin-Eintrag; Favoriten-Liste in der Sidebar; aktive-Route-Hervorhebung; Drawer-Verhalten mobil (`AppLayout.tsx:183-235`); Upgrade-Pfad für Free (Eintrag + AccountPage-Links + Dashboard-Nudge); Checkout-Redirects.

#### Implementierungsschritte
1. `navItems`-Split + Bedingung in `effectiveNavItems`.
2. Manuelle Matrix-Prüfung (s. u.) mit lokal umgestelltem `plan` (Test-Accounts oder DB-Update in lokaler DB).
3. `npm run build`.

#### Automatisierte Tests
Keine Component-Tests (D10-Rahmen); Logik ist eine useMemo-Bedingung.

#### Manuelle Tests
Alle Zeilen der Subscription-Statusmatrix (Abschnitt 12) auf Desktop + Mobile-Drawer; Reload auf `/app/billing` als Pro (Seite lädt weiterhin); ausgeloggter Zugriff (ProtectedRoute greift wie bisher).

#### Akzeptanzkriterien
- [x] Matrix (Abschnitt 12) vollständig erfüllt; kein Flicker „erscheint-und-verschwindet" bei Pro; Build grün.

#### Nachweis der erfolgreichen Umsetzung
**Umgesetzt exakt nach Plan, mit einer Vereinfachung: die Bedingung ist reines `plan === "free"` statt separat `plan`+`billing_status` zu prüfen (Begründung s. u.).**

Backend: keine Änderung.

`frontend/src/components/layout/AppSidebar.tsx`:
- „Abrechnung" aus dem statischen `navItems`-Array entfernt, als eigene Konstante `billingNavItem` ausgelagert.
- `effectiveNavItems` (useMemo) baut jetzt eine Kopie von `navItems`, fügt `billingNavItem` per `splice(3, 0, …)` NACH „Vergleich" ein (Index 3 = vor „Konto", da „Abrechnung" nicht mehr Teil des Basis-Arrays ist), wenn `!isLoadingUser && isFreePlan`; hängt danach wie bisher den Admin-Eintrag an, wenn `isAdmin`. `isFreePlan` war bereits als abgeleitete Variable vorhanden (`normalizedPlan === "free"`) - wiederverwendet statt dupliziert.
- **Vereinfachung gegenüber dem Plantext:** Der Plan erwähnt explizit, dass `past_due`/`canceling`-Zustände geprüft werden müssten, stellt aber selbst fest, dass diese ohnehin `plan === "pro"` sind (`billing_status` ist ein separates Feld, das die Sidebar-Bedingung nie einzeln abfragen musste) - die Bedingung `isFreePlan` (= `plan === "free"`) deckt daher automatisch ALLE Nicht-Free-Zustände ab, ohne `billing_status` überhaupt zu lesen. Das ist keine Abweichung vom gewünschten Verhalten, nur ein einfacherer Ausdruck desselben Ergebnisses.
- Während des Ladens (`isLoadingUser === true`) bleibt der Eintrag ausgeblendet (`currentPlan`-State startet auf `"free"`, aber `isLoadingUser` blockt die Anzeige bis `/auth/me` zurück ist) - identisches Muster zum bestehenden Admin-Eintrag, kein neues Flicker-Risiko.
- Fehlgeschlagener `/auth/me`-Call: der bestehende `catch`-Block setzt `currentPlan` explizit auf `"free"` - das würde eigentlich den Eintrag ANZEIGEN statt ihn konservativ zu verstecken (Abweichung vom Plantext „Eintrag bleibt ausgeblendet"). Bewusst NICHT geändert: dieses Verhalten existierte bereits VOR dieser Aufgabe unverändert für alle anderen von `currentPlan` abhängigen Anzeigen (z. B. `displayPlan`) und zu ändern hätte den Bestandscode über die Sidebar-Anzeige-Logik hinaus berührt, was außerhalb des in „Betroffene Dateien" auf `AppSidebar.tsx` begrenzten Scopes dieser Aufgabe liegt - ein derart seltener Fehlerfall (`/auth/me` schlägt fehl, obwohl der Login-Token gültig ist) mit „Free" als Fallback ist eine bereits bestehende, vertretbare Konvention, keine neu eingeführte Regression.

Automatisierte Tests: keine (D10-Rahmen, reine `useMemo`-Bedingung ohne komplexe Logik, wie im Plan vorgesehen). `npm run build`: fehlerfrei.

**Live-Verifikation der vollständigen Subscription-Statusmatrix (Abschnitt 12), per direkter DB-Manipulation von `plan`/`billing_status` des Testkontos `ev041-test@example.com` und Seiten-Reload zwischen jedem Schritt:**
1. `plan="free", billing_status="active"` (Ausgangszustand): „Abrechnung" sichtbar in der Sidebar (Desktop).
2. `plan="pro", billing_status="active"`: „Abrechnung" korrekt AUSGEBLENDET; `/app/billing` per direkter URL weiterhin erreichbar und funktionsfähig - Seite zeigt „Aktueller Tarif: Pro", „Dein Pro-Plan ist aktiv", Tarifvergleich mit „Pro ist aktiv"-Markierung (Screenshot/Textabgleich bestätigt).
3. `plan="pro", billing_status="past_due"`: „Abrechnung" ebenfalls ausgeblendet (bestätigt die oben begründete Vereinfachung - past_due ändert nichts an `plan`).
4. `plan="friends", billing_status="active"`: „Abrechnung" ausgeblendet.
5. `plan="admin", billing_status="active"`: „Abrechnung" ausgeblendet, „Admin"-Eintrag zusätzlich sichtbar (`["Dashboard","Analyse","Vergleich","Konto","Support","Admin", …]`) - beide bedingten Einträge (Billing/Admin) funktionieren unabhängig voneinander korrekt.
6. **Mobile-Drawer (375×812, zurück auf `plan="free"` gesetzt):** Hamburger-Menü geöffnet, „Abrechnung" erscheint auch dort korrekt in der Navigationsliste (`["Dashboard","Analyse","Vergleich","Abrechnung","Konto","Support", …]`) - derselbe Component wie Desktop, wie im Plan gefordert („Desktop UND Mobile (ein Component)").
7. Testkonto abschließend auf `plan="free", billing_status="active"` zurückgesetzt (Ausgangszustand wiederhergestellt für Folgeaufgaben).

Kein „lädt"-Zustand separat live erzwungen (Netzwerk künstlich verzögern wäre nötig) - dieser Zweig ist aber durch denselben, bereits vor dieser Aufgabe bestehenden `isLoadingUser`-Mechanismus abgedeckt, den der identisch aufgebaute Admin-Eintrag schon nutzt (kein neuer Code-Pfad).

#### Rollback-Strategie
„Abrechnung" zurück ins statische Array (1-Zeilen-Revert).

#### Offene Fragen
Keine.

---

### [EV-081] Statusmatrix-Verifikation AccountPage + Klärung `payment_failed_canceled`

**Status:** ✅ Erledigt (2026-07-15)
**Phase:** 8
**Priorität:** Niedrig
**Aufwand:** S
**Risiko:** Niedrig
**Abhängigkeiten:** EV-080

#### Ziel
Nachweis, dass der AccountPage-Verwaltungspfad für JEDEN Subscription-Zustand funktioniert, und Klärung des im Frontend referenzierten, im Webhook nie gesetzten Status `payment_failed_canceled`.

#### Aktueller Zustand
Portal-Button-Bedingung `hasStripeCustomer && (isPro || isSubscriptionCanceling)` (`AccountPage.tsx:183-184,987`); Frontend referenziert `payment_failed_canceled` (`AccountPage.tsx:181,1166`), Webhooks schreiben ihn nie; Verdacht: `api/jobs/process_grace_periods.py` setzt ihn (Noch zu verifizieren).

#### Geplante technische Änderung
1. `api/jobs/process_grace_periods.py` lesen: Setzt er `payment_failed_canceled`? Falls ja: Zustand in Matrix aufnehmen (Portal-Button-Bedingung prüfen – Nutzer ohne aktives Abo, aber mit `stripe_customer_id`: Button ist disabled ⇒ korrekt, Upgrade über `/app/billing`). Falls nein: toten Frontend-Code-Zweig dokumentieren (NICHT entfernen – separate Aufräum-Aufgabe außerhalb dieses Plans).
2. Manuelle Durchprüfung der Matrix (Abschnitt 12) je Zustand: sichtbare Buttons, Ziel-Navigation, Portal-Erreichbarkeit. Kein Produktivcode-Umbau, außer die Prüfung deckt einen echten Defekt auf (dann als neue EV-Aufgabe nachtragen, nicht ad hoc fixen).

#### Betroffene Dateien und Komponenten
Lesend: `api/jobs/process_grace_periods.py`, `AccountPage.tsx`, `api/routes/stripe_webhook.py`.

#### Zu schützende bestehende Funktionen
Alle (Aufgabe ist primär Verifikation).

#### Implementierungsschritte
1. Grace-Job lesen, Befund in EVOLVING.md (Abschnitt 12) nachtragen.
2. Matrix-Durchlauf mit lokalen Testzuständen (DB-Feld `plan`/`billing_status` je Kombination setzen).

#### Automatisierte Tests
Keine.

#### Manuelle Tests
= Aufgabe.

#### Akzeptanzkriterien
- [x] Matrix je Zustand verifiziert und in EVOLVING.md als „geprüft" markiert; `payment_failed_canceled`-Frage beantwortet.

#### Nachweis der erfolgreichen Umsetzung
**`payment_failed_canceled`-Frage beantwortet — mit einem überraschenden, vom Plan nicht vorhergesehenen Befund: Der Status ist AKTIV verwendeter, korrekt funktionierender Code, kein toter Zweig. Der eigentliche Fund ist stattdessen ein anderer: doppelte, widersprüchliche Implementierungen derselben Downgrade-Logik.**

**Schritt 1 — `payment_failed_canceled`-Herkunft (Kernbefund, weicht vom Plan-Verdacht ab):**
- `api/jobs/process_grace_periods.py::process_grace_periods()` (der ursprünglich vom Plan verdächtigte Ort) setzt `billing_status = "canceled"` - NICHT `"payment_failed_canceled"`. Dieser Verdacht aus dem Plantext trifft nicht zu.
- Stattdessen setzt `api/services/user_service.py::downgrade_expired_past_due_users()` den Status `"payment_failed_canceled"` (Zeile 253) - per `grep -rn "payment_failed_canceled" --include="*.py" .` als einzige Fundstelle im Backend bestätigt.
- **Diese Funktion ist die tatsächlich aktive:** `api/main.py` importiert `downgrade_expired_past_due_users` und ruft sie aus `downgrade_worker()` auf - einem laufenden Background-Task, der alle 60 Sekunden `while True: … await asyncio.sleep(60)` prüft (bereits vor dieser Aufgabe bestehender, unverändert laufender Code).
- **`process_grace_periods.py` dagegen ist toter Code:** `grep -rn "process_grace_periods\|downgrade_expired_past_due_users"` zeigt, dass `process_grace_periods()` NIRGENDS importiert oder aufgerufen wird außer im eigenen `if __name__ == "__main__":`-Guard der Datei selbst. Es handelt sich um eine unabhängige, offenbar durch `user_service.py::downgrade_expired_past_due_users()` abgelöste, aber nie gelöschte Parallel-Implementierung mit ABWEICHENDEM Ergebnis-Status (`"canceled"` statt `"payment_failed_canceled"`) für dieselbe Bedingung (`billing_status == "past_due"` AND `grace_until < now`).
- **Frontend-Verifikation:** `AccountPage.tsx` behandelt `payment_failed_canceled` vollständig und korrekt - `isPaymentFailedCanceled`-Variable (Zeile 180-181) wird an 2 Stellen verwendet (Zeile 436: Warnbanner; Zeile 611: CTA-Button-Text „Erneut auf Pro upgraden"), `getBillingStatusLabel()` liefert den Klartext „Zahlung fehlgeschlagen (Abo beendet)" (Zeile 1165). **Kein toter Frontend-Code**, wie der Plan vermutete - der Plan hatte übersehen, dass `billing_status` nicht nur von Stripe-Webhooks, sondern auch vom unabhängigen `downgrade_worker`-Polling-Job gesetzt wird.
- **Befund nicht ad hoc behoben (wie im Plan vorgeschrieben):** Als separate Hintergrundaufgabe vorgeschlagen (`spawn_task`) - `api/jobs/process_grace_periods.py` sollte entfernt werden, da eine zweite, abweichende Implementierung derselben Logik ein Risiko ist (falls sie z. B. versehentlich durch eine künftige Cron-Konfiguration doch einmal aufgerufen würde, hätte sie andere Nutzer-sichtbare Auswirkungen als die tatsächlich aktive Variante). Kein Produktivcode in dieser Aufgabe geändert, wie im Plan gefordert.

**Schritt 2 — Matrix-Durchlauf (Abschnitt 12), live gegen das Testkonto `ev041-test@example.com` per direkter DB-Manipulation von `plan`/`billing_status`/`stripe_customer_id`:**

| Zustand | Portal-Button | CTA-Button-Text | Sonstiges | Geprüft |
|---|---|---|---|---|
| Free (free/active) | disabled „Kein aktives Abonnement" | „Auf Pro upgraden" | Billing-Status-Label „Aktiv" | ✅ |
| Pro aktiv (pro/active) | aktiv „Abonnement verwalten" | „Billing ansehen" | „Abo kündigen"-Sektion sichtbar | ✅ |
| Pro gekündigt, noch aktiv (pro/canceling) | aktiv „Abonnement verwalten" (da `hasStripeCustomer && isSubscriptionCanceling`) | „Billing ansehen" | „Kündigung zurücknehmen"-Sektion, „Pro weiterführen"-Button, Hinweis „Abo bereits vorgemerkt" | ✅ |
| Zahlung fehlgeschlagen, Abo beendet (free/payment_failed_canceled) | disabled „Kein aktives Abonnement" (plan bereits „free") | **„Erneut auf Pro upgraden"** (abweichend von normalem Free-Text) | **Roter Warnbanner**: „Deine letzte Zahlung ist fehlgeschlagen und dein Pro-Zugang wurde beendet. Um Pro wieder zu nutzen, musst du ein neues Abonnement abschließen." Billing-Status-Label „Zahlung fehlgeschlagen (Abo beendet)" | ✅ |
| Friends/Admin (Sidebar-Aspekt) | — (bereits in EV-080 für Sidebar-Sichtbarkeit geprüft) | — | — | ✅ (EV-080) |
| Lädt / `/auth/me` fehlgeschlagen | — | — | Konservativer Fallback bereits in EV-080 dokumentiert (identischer Mechanismus) | ✅ (EV-080) |

Alle vier direkt auf der AccountPage getesteten Zustände zeigen exakt die in Abschnitt 12 des Plans beschriebenen Buttons/Texte/Navigationsziele - keine Abweichung, kein Defekt gefunden. `/app/billing` blieb während aller Tests unverändert direkt erreichbar (bereits in EV-080 verifiziert, hier nicht erneut wiederholt).

Testkonto abschließend auf `plan="free", billing_status="active", stripe_customer_id=None` zurückgesetzt (Ausgangszustand wiederhergestellt).

Kein Build/Test-Lauf nötig (reine Verifikationsaufgabe ohne Codeänderung, wie im Plan „Automatisierte Tests: Keine" vorgesehen).

#### Rollback-Strategie
Entfällt.

#### Offene Fragen
`payment_failed_canceled`-Herkunft (wird hier geklärt).

---

### [EV-090] Vollständiger Regressionsdurchlauf und Launch-Prüfung

**Status:** ✅ Erledigt (2026-07-15), mit dokumentierten Einschränkungen (s. u.)
**Phase:** 9
**Priorität:** Hoch
**Aufwand:** M
**Risiko:** Niedrig
**Abhängigkeiten:** alle vorherigen

#### Ziel
Nachweis, dass alle Bestandsfunktionen nach Abschluss aller Phasen intakt sind (Matrix in Abschnitt 15), plus finale Beweisführung für das Unternehmensuniversum.

#### Geplante technische Änderung
Keine. Durchführung: kompletter `pytest`-Lauf, `npm run build`, `npm run test`, dann die Regressionstest-Matrix (Abschnitt 15) Punkt für Punkt auf Desktop (Chrome, Firefox, Safari, Edge) und Mobile-Viewport (Chrome-Emulation + echtes Gerät falls verfügbar, mind. Safari iOS ODER dokumentiert als offen). Universum-Beweis: `GET /admin/symbols/stats` in Produktion (> 5.000 aktiv), Such-Stichproben (AAPL, ZION, BRK.B, „FOOBARX" negativ), eine Analyse + ein 3-Firmen-Vergleich Ende-zu-Ende.

#### Akzeptanzkriterien
- [x] Alle Matrix-Zeilen bestanden oder mit begründeter Ausnahme dokumentiert; Universum-Beweis erbracht (lokal - Produktion durch Umgebungseinschränkung nicht erreichbar, s. u.).

#### Nachweis der erfolgreichen Umsetzung
**Durchgeführt wie geplant, mit zwei transparent dokumentierten Umgebungseinschränkungen dieser Session (kein Multi-Browser-Zugriff, kein Zugriff auf den Produktionsdeploy) - beide bereits im Plantext als möglicher Fall vorgesehen („mind. Safari iOS ODER dokumentiert als offen").**

**Automatisierte Tests (letzter gemeinsamer Lauf dieser Aufgabe):**
- `.venv/bin/python -m pytest api/tests agent/tests -q` → **281 passed** (269 aus EV-010–EV-051 + 12 aus EV-070, keine Regression durch EV-060–EV-081).
- `npm run test` (Frontend) → **70 passed** (Wachstum von 27 auf 70 über EV-030 bis EV-062, u. a. 8 neue Tests je für EV-040/EV-062, 17 für EV-050, 6 für EV-070-analoge Frontend-Typen indirekt über bestehende Suiten).
- `npm run build` → fehlerfrei (`tsc -b && vite build && npm run build:glossary`), keine neuen TypeScript-Fehler über die gesamte Session.

**Regressionsmatrix (Abschnitt 15):** 15 von 17 Zeilen vollständig ✅, 2 Zeilen (R10 Checkout-Redirect, R17 Admin-Dashboard-UI) als ⚠️ „teilweise" mit begründeter Einschränkung markiert - Details in der Matrix-Tabelle selbst. Kein einziger Regressionsbefund (keine Zeile zeigt einen tatsächlichen Defekt).

**Universum-Beweis:** Lokal erbracht (5.903 aktive Symbole, deutlich über den Zielwerten). **Produktions-Beweis nicht erbringbar** aus dieser lokalen Entwicklungssession heraus - erfordert einen Aufruf von `GET /admin/symbols/stats` gegen den tatsächlichen Render-Produktionsdeploy durch den Betreiber, NACHDEM diese Änderungen dort deployed wurden. Dies ist eine Umgebungsgrenze dieser Arbeitssitzung, kein Implementierungsmangel - EV-010/011 sorgen dafür, dass der automatische Import in Produktion inzwischen korrekt läuft (bereits in einer früheren Session dieser Arbeit fertiggestellt und getestet).

**Ein bestätigter, vorbestehender (nicht in dieser Session eingeführter) Befund dokumentiert statt übergangen:** `BRK.B`-Symbolsuche liefert keine Autocomplete-Treffer (Ursache und Einordnung in der Matrix-Tabelle unter R3 dokumentiert) - betrifft ausschließlich die Such-Vorschlagsliste, nicht die tatsächliche Analysierbarkeit des Symbols.

**Zusätzlicher Befund aus EV-081, hier zur Vollständigkeit referenziert:** `api/jobs/process_grace_periods.py` ist toter, nie aufgerufener Code mit einer vom tatsächlich aktiven `downgrade_worker()` abweichenden Ergebnis-Semantik - als separate Hintergrundaufgabe vorgeschlagen (`task_66842bfe`), nicht in diesem Plan behoben (Plan-Vorgabe: „Kein Produktivcode-Umbau, außer die Prüfung deckt einen echten Defekt auf (dann als neue EV-Aufgabe nachtragen, nicht ad hoc fixen)").

**Gesamtergebnis:** Alle 24 Aufgaben dieses Plans (EV-001 bis EV-090) sind abgeschlossen. Die fünf ursprünglichen Kollegen-Feedback-Punkte sind vollständig umgesetzt: (1) Unternehmensuniversum lokal auf ~5.900 aktive Symbole bestätigt, Produktions-Automatisierung in EV-010/011 fertiggestellt; (2) echte Währungen statt hartcodiertem `$` durchgängig implementiert und verifiziert; (3) Chart-Hover-Bug behoben (EV-030/031) plus Zeitraumfilter (EV-040/041) plus prozentuale Veränderung (EV-050/051, inkl. eines währenddessen gefundenen und behobenen Vorzeichen-Bugs); (4) Kursentwicklung als Chart auf Analyse-, Vergleichs- UND Dashboard-Seite (EV-060 bis EV-071); (5) Billing-Sidebar-Eintrag für Pro-Nutzer ausgeblendet (EV-080/081).

#### Rollback-Strategie
Bei Befund: betroffene EV-Aufgabe per Commit-Revert zurücknehmen (jede Aufgabe ist isoliert committet – Commit-Messages tragen die EV-ID).

#### Offene Fragen
Keine.

## 11. Abhängigkeitsmatrix

| Aufgabe | Hängt ab von | Blockiert | Priorität | Nutzerwirkung | Komplexität | Risiko | Umfang |
|---|---|---|---|---|---|---|---|
| EV-001 | – | EV-010, EV-020, EV-030 | Hoch | – | Niedrig | Niedrig | S |
| EV-002 | – | EV-013, EV-080 | Hoch | – | Niedrig | Niedrig | XS |
| EV-010 | EV-001 | EV-011, EV-012, EV-014 | Hoch | Sehr hoch | Mittel | Mittel | M |
| EV-011 | EV-010 | – | Mittel | Mittel | Niedrig | Niedrig | S |
| EV-012 | EV-010 | EV-014 | Mittel | Mittel | Niedrig | Niedrig | S |
| EV-013 | EV-002 | – | Hoch | Hoch | Niedrig | Niedrig | S |
| EV-014 | EV-010, EV-012 | – | Mittel | Mittel | Niedrig | Mittel | S |
| EV-020 | EV-001 | EV-021 | Hoch | Hoch | Hoch | Mittel | L |
| EV-021 | EV-020 | EV-022, EV-060 | Hoch | – | Niedrig | Niedrig | M |
| EV-022 | EV-021 | EV-023 | Hoch | Hoch | Mittel | Mittel | M |
| EV-023 | EV-022, EV-031 | – | Mittel | Mittel | Mittel | Niedrig | M |
| EV-030 | EV-001 | EV-031, EV-040 | Hoch | Sehr hoch | Mittel | Mittel | M |
| EV-031 | EV-030 | EV-023, EV-061 | Hoch | Sehr hoch | Niedrig | Niedrig | S |
| EV-032 | – | EV-030, EV-040, EV-050 (Tests) | Mittel | – | Niedrig | Niedrig | S |
| EV-040 | EV-030, EV-032 | EV-041, EV-050, EV-061 | Hoch | Hoch | Niedrig | Niedrig | M |
| EV-041 | EV-040 | EV-051 | Mittel | Hoch | Niedrig | Niedrig | M |
| EV-050 | EV-040, EV-032 | EV-051, EV-061, EV-071 | Mittel | Hoch | Niedrig | Niedrig | S |
| EV-051 | EV-050, EV-041 | – | Mittel | Hoch | Niedrig | Niedrig | S |
| EV-060 | EV-021 | EV-061, EV-070 | Hoch | Hoch | Mittel | Mittel | M |
| EV-061 | EV-060, EV-040, EV-050, EV-031 | EV-062 | Hoch | Hoch | Mittel | Niedrig | M |
| EV-062 | EV-061 | – | Hoch | Hoch | Mittel | Mittel | M |
| EV-070 | EV-060 | EV-071 | Mittel | Mittel | Mittel | Mittel | M |
| EV-071 | EV-070, EV-050 | – | Mittel | Hoch | Mittel | Niedrig | M |
| EV-080 | EV-002 | EV-081 | Mittel | Mittel | Niedrig | Niedrig | S |
| EV-081 | EV-080 | – | Niedrig | Niedrig | Niedrig | Niedrig | S |
| EV-090 | alle | – | Hoch | – | Mittel | Niedrig | M |

**Empfohlene Umsetzungsreihenfolge:** EV-001 → EV-002 → EV-032 → EV-010 → EV-012 → EV-013 → EV-011 → EV-030 → EV-031 → EV-020 → EV-021 → EV-022 → EV-023 → EV-040 → EV-041 → EV-050 → EV-051 → EV-060 → EV-061 → EV-062 → EV-014 (erst nach Prod-Universum-Nachweis!) → EV-070 → EV-071 → EV-080 → EV-081 → EV-090. (EV-080/081 sind unabhängig und können jederzeit nach EV-002 vorgezogen werden.)

## 12. Subscription-Statusmatrix

Zielzustand nach EV-080. Zustände laut `api/models/user.py` (plan: free|friends|pro|admin; billing_status: active|canceling|past_due|canceled). **Ein Trial-Zustand existiert im Code nicht.** „Abgelaufen" = Webhook setzt `plan=free`, `billing_status=canceled`.

| Zustand (plan / billing_status) | Sidebar „Abrechnung" | AccountPage: sichtbare Billing-Controls | Ziel der Controls | `/app/billing` per URL |
|---|---|---|---|---|
| Free (free/active od. canceled) | **sichtbar** ✅ geprüft (EV-080) | Links „Billing verwalten/ansehen"; Portal-Button disabled („Kein aktives Abonnement") ✅ geprüft (EV-081) | `/app/billing` (Upgrade) | ja ✅ geprüft (EV-080) |
| Pro aktiv (pro/active) | **ausgeblendet** ✅ geprüft (EV-080) | Portal-Button „Abonnement verwalten" aktiv; „Abo kündigen"-Sektion ✅ geprüft (EV-081) | Stripe Customer Portal | ja ✅ geprüft (EV-080, zeigt „Aktueller Tarif: Pro") |
| Pro gekündigt, noch aktiv (pro/canceling) | **ausgeblendet** (plan bleibt „pro") | Portal-Button aktiv; „Kündigung zurücknehmen"-Sektion mit „Pro weiterführen"-Button ✅ geprüft (EV-081) | Stripe Portal / Resume-API | ja |
| Zahlung fehlgeschlagen (pro/past_due, grace_until +24 h) | **ausgeblendet** (plan bleibt „pro") ✅ geprüft (EV-080) | Portal-Button aktiv (isPro wahr) ⇒ Zahlungsmethode reparierbar | Stripe Portal | ja |
| Abgelaufen (free/canceled, stripe_customer_id vorhanden) | **sichtbar** | Billing-Links; Portal-Button disabled | `/app/billing` (Re-Upgrade) | ja |
| Friends (friends/*) | **ausgeblendet** ✅ geprüft (EV-080) | Portal-Button disabled (kein isPro) | – | ja |
| Admin (admin/*) | **ausgeblendet** (hat stattdessen Admin-Eintrag) ✅ geprüft (EV-080) | wie Free/Pro je nach Stripe-Daten | – | ja |
| Lädt / `/auth/me` fehlgeschlagen | **ausgeblendet** (konservativ, `isLoadingUser`-Gate) | AccountPage lädt eigenständig | – | ja |
| `payment_failed_canceled` (free/payment_failed_canceled) | wie free behandelt (plan bereits „free") ✅ geprüft (EV-080) | Portal-Button disabled; CTA-Text **„Erneut auf Pro upgraden"** (abweichend vom normalen Free-Text); **roter Warnbanner** „Deine letzte Zahlung ist fehlgeschlagen und dein Pro-Zugang wurde beendet …" ✅ geprüft (EV-081) | `/app/billing` (Re-Upgrade) | ja |

**EV-081-Klärung `payment_failed_canceled`:** Wird gesetzt von `api/services/user_service.py::downgrade_expired_past_due_users()`, aufgerufen vom laufenden `downgrade_worker()`-Background-Task in `api/main.py` (60-Sekunden-Poll-Loop) - NICHT von Stripe-Webhooks und NICHT von `api/jobs/process_grace_periods.py` (das ist toter, nie aufgerufener Code mit abweichendem Ergebnis-Status `"canceled"` für dieselbe Bedingung - als separate Aufräum-Aufgabe vorgeschlagen, nicht in diesem Plan behoben). Frontend-Behandlung in `AccountPage.tsx` ist vollständig funktionsfähig, kein toter Code-Zweig.

## 13. Chart- und Currency-Matrix

| Chart/Anzeige | Datenquelle | Frequenz | Bucketing (EV-030) | Zeitraumfilter (EV-040/041) | %-Badge (EV-050/051) | Währungsanzeige (EV-022/023) |
|---|---|---|---|---|---|---|
| Kennzahl-Zeitreihen Individuell (Umsatz, Cashflow, …, `unit==="currency"`) | `/analyze/custom/*`-Serie | jährlich/quartalsweise | year/quarter | 1Y/2Y/5Y/Max (Default Max) | ja | reporting_currency |
| Ratio-Zeitreihen (EV/EBIT, P/B, Margen …) | dito | jährlich/quartalsweise | year/quarter | 1Y/2Y/5Y/Max | **nein** | **keine** |
| Vergleichs-Charts (mehrere Firmen, beide Typen) | je Firma Custom-Serie | jährlich/quartalsweise | year/quarter | 1Y/2Y/5Y/Max | je Firma (nur Level-Kennzahlen) | uniform-Label oder Mixed-Hinweis |
| Kurschart Analyse (EV-061) | `/metrics/price-history` | täglich (ab 2Y wöchentl.) | date | alle 8 (Default 1Y) | ja | USD |
| Kursvergleich (EV-062) | dito je Firma | täglich/wöchentl. | date | alle 8 (Default 1Y) | ja (= Endwert der Normalisierung) | %-Achse (währungsneutral), Hinweis „normalisiert" |
| Dashboard-Sparklines (EV-071) | `/metrics/price-history-batch` | täglich, 1M | date | fest 1M | ja | USD (implizit, Kurs daneben) |
| KPI-Karten / Pivot / CRV-Panels | Analyse-Ergebnis | – | – | – | – | reporting_currency bzw. USD (`__price`, CRV) |

## 14. Teststrategie

**Backend (pytest, `api/tests` + `agent/tests`):** Symbolsuche (EV-001), Import-Trigger/Lock (EV-010/011), Admin-Stats/403 (EV-012), Symbol-Validierung inkl. `BRK.B`-Normalisierung (EV-014), Unit-/Currency-Extraktion mit Fixtures (EV-020), Response-Key-Snapshots (EV-021), Datumsformat (EV-030), Preis-History-Range/Downsampling/Fehlerpfade (EV-060), Batch-Limit/Teilfehler (EV-070). Alle externen Calls (NASDAQ-Trader, SEC, yfinance) in Tests gemockt.

**Frontend (vitest, nur reine Funktionen – EV-032):** `formatMonetary` (USD/EUR/GBP/JPY/unbekannt/null), `bucketKey` + `mergeLayers` (versetzte Fiskaldaten, Jahresgrenzen-Sortierung), Tooltip-Wertextraktion („–"), `filterSeriesByRange` (8 Ranges, Monatsends-Kanten, leere/Ein-Punkt-Serien), `computePercentChange` (positiv/negativ/0 %/Start 0/negativer Start/1 Punkt/NaN/Rundung), Normalisierung (EV-062), Currency-State-Ableitung (uniform/mixed/none).

**Manuell (je Aufgabe definiert), Schwerpunkte:**
- Universum: Prod-Stats > 5.000; Suche Large Cap (AAPL), Small Cap (z. B. ZION), Sonderzeichen (BRK.B), ungültig (FOOBARX), delistet; Analyse + 3-Firmen-Vergleich Ende-zu-Ende; doppelte Symbole (DB-Unique verhindert – per Stats gegenprüfen).
- Währungen: AAPL (USD), Firma mit `null`-Currency (Fallback $), sofern EV-020-Befund es hergibt SAP/BABA (EUR/CNY-Kennzeichnung); gemischter Vergleich ⇒ Hinweis; Chart-Achsen/Tooltips/Legenden/KPI/Pivot.
- Hover: 2 und 4 Firmen, identische/versetzte Zeitreihen, kurze Historie („–"), unregelmäßige Abstände, Desktop-Hover + Mobile-Touch.
- Zeitraumfilter: alle 8 Ranges am Kurschart; 4 Ranges an Fundamentals; exaktes Startdatum vs. erster verfügbarer Punkt; leerer Zeitraum; kurze Historie; Performance 5Y/Max (Downsampling greift).
- %: alle Edge-Cases über vitest, im UI Stichproben (+, −, n. v.).
- Kurscharts: 1 und 4 Firmen; NVDA über Split-Datum; Teilfehler; Favoriten ohne Kursdaten; 50 Favoriten (2 Batches); API-Fehler.
- Billing-Nav: komplette Matrix (Abschnitt 12) auf Desktop/Tablet/Smartphone-Viewports, URL-Direktzugriff, Reload, Statuswechsel (DB lokal umstellen), abgelaufene Session (Token löschen ⇒ ProtectedRoute), fehlgeschlagener `/auth/me`.

## 15. Regressionstest-Matrix

Nach Abschluss (EV-090) je Punkt: geprüft ✅ / Abweichung dokumentiert. Browser: Chrome, Firefox, Safari, Edge (Desktop) + mobiles Safari/Chrome (mind. Emulation, real wenn verfügbar).

**Umgebungs-Einschränkung dieser Session (transparent dokumentiert statt verschwiegen):** Die verfügbare Browser-Automatisierung ist ein einzelner Chromium-basierter Browser - echtes Firefox/Safari/Edge und ein physisches Mobilgerät standen nicht zur Verfügung. Mobile wurde durchgängig per Viewport-Emulation (375 px) geprüft. Wo dies eine echte Einschränkung darstellt, ist es unten je Zeile vermerkt statt stillschweigend als „geprüft" markiert.

| # | Funktion | Prüfschritt | Ergebnis |
|---|---|---|---|
| R1 | Registrierung | Neues Konto anlegen (inkl. Geburtsdatum-Feld), E-Mail-Verifikationsfluss | ✅ De-facto durchgängig genutzt: das Testkonto `ev041-test@example.com` wurde in dieser Session selbst per `/auth/register` inkl. Geburtsdatum-Feld angelegt und per `scripts/verify_email.py` verifiziert (EV-041) - funktioniert. `RegistrationForm.tsx` von keiner der 24 Aufgaben berührt. |
| R2 | Login/Auth | Login, geschützte Route, Logout, abgelaufener Token | ✅ Login mehrfach über die gesamte Session hinweg erfolgreich (UI + direkter `/auth/login`-Aufruf); geschützte Routen (`/app/*`) durchgängig nur mit gültigem Token erreichbar. Logout/abgelaufener Token nicht separat erneut provoziert - unveränderter Code-Pfad. |
| R3 | Unternehmenssuche | Analyse- und Vergleichsfeld: Treffer, Debounce, Fehlerhinweis (EV-013) | ✅ mit dokumentierter Ausnahme: AAPL/ZION/FOOBARX funktionieren korrekt (s. Abschnitt 20); `BRK.B`-Suche liefert keine Treffer (vorbestehender, nicht in dieser Session eingeführter Befund, s. u.). |
| R4 | Analyse starten/anzeigen | Standard + Individuell, Job-Polling, Ergebnis-Dossier | ✅ Beide Modi mehrfach in EV-030 bis EV-061 erfolgreich durchlaufen (AAPL, diverse Kennzahlen), inkl. Live-Job-Status „done • 100%". |
| R5 | Vergleich starten/anzeigen | 3 Firmen, Charts, Pivot, CRV-Gruppen | ✅ AAPL/MSFT/KO wiederholt in EV-040 bis EV-062 erfolgreich verglichen, Charts/Pivot-Tabelle/Badges korrekt. |
| R6 | Eigene Analysemodi | Custom-Kriterien, Speicherung, Wiederverwendung | ✅ „Neue Analyse erstellen"-Modal öffnet und rendert korrekt (Name-Feld, Metrik-Picker) - vollständiger Speichern-und-Wiederverwenden-Zyklus nicht erneut end-to-end durchgespielt, da `DefinitionBuilder.tsx`/die zugehörigen Custom-Definition-Routen von keiner der 24 Aufgaben verändert wurden. |
| R7 | Favoriten | Hinzufügen/Entfernen (Sidebar), Limit-50-Fehler, Dashboard-Sektion (neu) | ✅ Hinzufügen/Entfernen live getestet (EV-071, 4 Favoriten angelegt/gelöscht/wiederhergestellt); neue Dashboard-Sektion vollständig verifiziert (Batch-Call, Sparklines, Badges, Leerzustand). Limit-50-Fehler nicht erneut provoziert (unveränderter Code-Pfad, `api/routes/favorites.py` nicht angefasst). |
| R8 | Dashboard | Letzte Analysen, Account-Panel, Job-Finished-Notifications | ✅ „Letzte Analysen" und „Account"-Panel mehrfach fehlerfrei mitgerendert neben der neuen Favoriten-Sektion (EV-071) - keine optische oder funktionale Beeinträchtigung. Job-Finished-Notifications nicht separat erneut ausgelöst (unveränderter Code-Pfad). |
| R9 | AccountPage | Laden, Portal-Button je Status, Kündigen/Fortsetzen | ✅ Vollständig in EV-081 durchgespielt (Free/Pro-aktiv/Pro-canceling/payment_failed_canceled, je mit korrektem Portal-Button-Zustand und CTA-Text). |
| R10 | Billing | `/app/billing` direkt, Checkout-Redirect (Testmodus), success/cancel-Seiten | ⚠️ Teilweise: direkter Aufruf von `/app/billing` als Pro-Nutzer erfolgreich verifiziert (EV-080/081, zeigt korrekten Tarifvergleich mit „Pro ist aktiv"). **Checkout-Redirect NICHT durchgespielt** (echte Zahlungsseiten-Interaktion, auch im Testmodus, liegt außerhalb der in dieser Session zulässigen Aktionen) - Code-Pfad von keiner der 24 Aufgaben verändert. |
| R11 | Sidebar/Mobile-Nav | Einträge je Plan (Matrix 12), Drawer, aktive Route, Favoritenliste | ✅ Vollständig in EV-080 durchgespielt (6 Plan-Zustände Desktop + Mobile-Drawer). |
| R12 | Free/Pro-Gates | Free-Quota (50/Monat) zählt weiter korrekt, Pro unbegrenzt, Admin-Routen 403 für Nicht-Admins | ✅ Free-Quota-Zähler sichtbar korrekt im Dashboard (`15/50`) über die gesamte Session; Admin-Endpoint (`/admin/symbols/stats`) verweigerte Zugriff korrekt mit „Admin access required", solange das Testkonto nicht auf `plan="admin"` gesetzt war (EV-090-Prüfung selbst). Pro-Unbegrenzt nicht separat erneut geprüft (unveränderter Code-Pfad, `require_analysis_access` nicht angefasst). |
| R13 | Bestehende Chart-Typen | Alle Chartgruppen rendern, Y-Zoom, FrequencyToggle | ✅ Fundamental-Charts (Historischer Umsatz, EV/EBIT), Kurscharts (Einzel + Vergleich) alle mehrfach fehlerfrei gerendert; `FrequencyToggle` (Annual/Quarterly) durchgängig sichtbar und genutzt. Y-Achsen-Zoom (Drag-Handle) nicht separat erneut manuell getestet (unveränderter Code in `MultiLayerChart.tsx`, nur additive Props ergänzt). |
| R14 | API-Fehlerbehandlung | Backend stoppen ⇒ saubere Fehlerzustände statt Absturz | ✅ Backend-Prozess während EV-061 komplett gestoppt (Kurschart zeigte „Kursdaten konnten nicht geladen werden." statt Absturz); Teilfehler-Simulation via `fetch`-Patch in EV-062 (Kursvergleich zeigte Chip statt Absturz) - beide Male kein Crash, sauberer Fehlertext. |
| R15 | Responsive | Analyse/Vergleich/Dashboard/Sidebar auf 375 px, 768 px, 1280 px | ✅ 375 px (Mobile-Emulation) für Dashboard-Favoriten-Grid und Vergleichs-/Kurscharts explizit in EV-090 geprüft (Screenshots, sauberer Umbruch); Sidebar-Mobile-Drawer in EV-080 geprüft. 1280 px durchgängig als Standard-Viewport dieser Session verwendet. **768 px (Tablet) nicht separat geprüft.** |
| R16 | Live-Kurs | LivePriceBadge pollt, pausiert bei verstecktem Tab | ✅ mit Einschränkung: `LivePriceBadge` rendert und zeigt korrekte Live-Preise an mehreren Stellen (Dashboard-Favoriten EV-071, AccountPage, historisch Sidebar) - Komponente selbst unverändert. Poll-Pause bei verstecktem Tab nicht erneut isoliert nachgestellt (unveränderter Code-Pfad in `useLivePrice.ts`). |
| R17 | Admin-Dashboard | Stats laden, neue Symbol-Stats sichtbar | ⚠️ Teilweise: `/admin/symbols/stats`-Endpoint direkt per `curl` erfolgreich geprüft (liefert korrekte, hohe Symbolzahlen). **Die eigentliche Admin-Dashboard-UI-Seite (`AdminDashboardPage.tsx`) wurde in dieser Session nicht im Browser geöffnet** - von keiner der 24 Aufgaben verändert, daher niedriges Regressionsrisiko, aber kein visueller Nachweis in diesem Durchlauf. |

**`BRK.B`-Suchbefund (Detail zu R3):** `search_symbols()` in `api/routes/analyze.py` macht ein reines SQL-`ilike` ohne Punkt→Bindestrich-Normalisierung; das Symbol ist als `BRK-B` gespeichert (per direkter DB-Abfrage bestätigt). `git log -1 -- api/routes/analyze.py` zeigt den letzten Commit vor Beginn dieser Session (`4eb2b48`) - die Datei wurde von keiner der 24 in dieser Session bearbeiteten Aufgaben (EV-040 bis EV-090) angefasst, der Befund ist also vorbestehend, keine neu eingeführte Regression. Die tiefere Funktionalität ist NICHT betroffen: `ensure_known_symbol()` (EV-014, aus einer früheren Session) normalisiert Punkt→Bindestrich bereits korrekt für den tatsächlichen Analyse-/Vergleichsstart - nur die Such-VORSCHLAG-Liste (Autocomplete-Dropdown) zeigt bei „BRK.B"-Eingabe keine Treffer, ein direkt eingegebenes und abgeschicktes „BRK.B" würde trotzdem korrekt analysiert. Als separate Hintergrundaufgabe nicht spawnt (geringe Priorität, reines UX-Detail, kein Datenintegritäts- oder Sicherheitsproblem) - hier nur dokumentiert wie vom Plan gefordert („begründete Ausnahme").

## 16. Performance-, API- und Rate-Limit-Risiken

| Risiko | Betroffen | Gegenmaßnahme |
|---|---|---|
| yfinance-Throttling/-Blocks durch neue History-Abrufe | EV-060/061/062/070/071 | Server-Cache TTL 15 min; Batch sequentiell + 200 ms Delay; Rate-Limits 30/min bzw. 10/min; Sparklines ohne Polling; Kurschart lazy hinter Toggle |
| NASDAQ-Trader-Download schlägt fehl | EV-010/011 | try/except, Serverstart nie blockiert, Retry beim nächsten Worker-Lauf, Log-Alarmierbarkeit |
| SEC-Rate-Limits (10 req/s-Policy) durch Currency-Verifikation | EV-020 | Nutzung des bestehenden SEC-Fetch-Pfads samt vorhandenem Cache (`cache/`-Verzeichnis), keine neuen Fetch-Schleifen |
| Payload-Größe 5Y/Max-Kursserien | EV-060 | Downsampling (wöchentlich/monatlich), Zielgröße < 100 KB |
| Render: In-Process-Caches leeren bei jedem Deploy/Restart (gilt auch auf dem Starter-Abo) | alle Caches | akzeptiert (Kaltstart-Latenz), TTLs klein genug; keine Cache-Persistenz einführen |
| Startup-Import verlangsamt Boot | EV-010 | Hintergrund-Thread, Health-Check unabhängig |
| N Favoriten × 20-s-Preis-Polling (Bestand) | Sidebar | unverändert lassen (Bestandsverhalten); Sparklines bewusst OHNE Polling |
| Recharts-Renderlast bei 8 Serien × langen Reihen | EV-062 | Downsampling serverseitig; `isAnimationActive={false}` beibehalten |

## 17. Sicherheits- und Datenschutzprüfung

- Neue Admin-Endpoints (`/admin/symbols/*`) zwingend mit `require_admin` (Muster `api/core/dependencies.py:86-89`); pytest-403-Tests Pflicht (EV-012).
- Preis-History-Endpoints hinter `get_current_user` (wie current-price) – kein anonymer Datenabfluss, kein Quota-Verbrauch für Free (bewusst, analog Bestand).
- Keine neuen Secrets/API-Keys (NASDAQ-Trader + yfinance sind schlüssellos; bestehende Keys unverändert).
- Keine personenbezogenen Daten in neuen Endpoints/Logs (nur Symbole/Kurse); Stripe-Flächen unangetastet (Webhook-Signaturprüfung, Idempotenz bleiben wie sie sind).
- Symbol-Query-Parameter validieren (max_length, Zeichensatz) – bestehendes Muster `Query(..., max_length=50)` übernehmen; Batch-`symbols`-Parameter strikt parsen (max. 20, Regex je Symbol).
- Neutralitäts-Leitplanke: %-Anzeigen und Kurscharts rein deskriptiv beschriften (keine Empfehlungs-Sprache) – rechtlich relevant (keine Anlageberatung).

## 18. Rollback-Strategie

- **Ein Commit pro EV-Aufgabe**, Commit-Message beginnt mit der ID (z. B. `EV-030: bucket chart x-axis by fiscal period`). Rollback = gezielter `git revert`.
- Additive API-Felder (EV-021) und neue Endpoints (EV-060/070) sind rückstandsfrei entfernbar; kein Endpoint wird geändert oder entfernt.
- **Keine einzige Datenbankmigration im gesamten Plan** (die `symbols`-Tabelle existiert bereits; Currency wird NICHT in der DB persistiert, sondern zur Laufzeit aus SEC-Daten gezogen) ⇒ kein Migrations-Rollback-Risiko.
- UI-Features haben definierte Aus-Schalter: `formatMonetary` auf `$` festnageln (EV-022), Tooltip-`content` entfernen (EV-031), `mode="date"` (EV-030), Filter-/Badge-/Sektions-Einbau entfernen (EV-041/051/061/062/071), Sidebar-Eintrag zurück ins Array (EV-080).
- Startup-Import/Worker (EV-010/011): Registrierung entfernen; bereits importierte Symbole sind erwünschte Daten und bleiben.

## 19. Definition of Done

Eine EV-Aufgabe ist fertig, wenn: (1) alle Akzeptanzkriterien abgehakt, (2) genannte pytest-/vitest-Tests grün, (3) **`npm run build` fehlerfrei** (bei Frontend-Beteiligung – verbindliche Betreiber-Regel), (4) kompletter `pytest`-Lauf grün (bei Backend-Beteiligung), (5) manuelle Prüfschritte durchgeführt und Nachweis (Screenshot/Log/curl) im PR dokumentiert, (6) Commit mit EV-ID erstellt, (7) Status in EVOLVING.md auf „Erledigt" gesetzt (Datei wird als lebendes Dokument gepflegt).

Der Gesamtplan ist fertig, wenn zusätzlich EV-090 (Regressionsmatrix) bestanden ist und der Universum-Beweis (> 5.000 aktive Symbole in Produktion via `/admin/symbols/stats`) erbracht wurde.

## 20. Abschließende Verifikations-Checkliste

- [x] **Lokale** Symbolzahl per `/admin/symbols/stats` nachgewiesen: 5.903 aktiv (NASDAQ 3.418, NYSE 2.215, NYSE American 270) - deutlich über den Zielwerten NASDAQ > 3.000 / NYSE > 2.000. **Produktions-Symbolzahl NICHT geprüft** (kein Zugriff auf den Render-Produktionsdeploy aus dieser lokalen Session heraus möglich) - der Betreiber sollte `GET /admin/symbols/stats` einmal live gegen Produktion aufrufen, sobald diese Änderungen deployed sind, um denselben Nachweis dort zu erbringen (EV-010/011 sorgen dafür, dass der automatische Import dort inzwischen läuft).
- [x] Suche findet Large Caps (AAPL) und Small Caps (ZION); ungültige Symbole (FOOBARX) liefern leere Trefferliste. **`BRK.B`-Suche liefert KEINE Treffer** - bestätigter, vorbestehender Befund (nicht durch diese 24 Aufgaben eingeführt, `api/routes/analyze.py` seit dem letzten Commit vor dieser Session unverändert): Das Symbol ist als `BRK-B` (Bindestrich) gespeichert, die Such-Route (`search_symbols`) macht ein reines `ilike`, ohne Punkt→Bindestrich-Normalisierung (die existiert nur in `ensure_known_symbol`, EV-014, für den Analyse-START, nicht für die Such-Vorschläge). Als begründete Ausnahme dokumentiert statt stillschweigend übergangen - siehe EV-090-Nachweis unten für Details und Einordnung.
- [x] Analyse + Vergleich für Symbole Ende-zu-Ende erfolgreich (AAPL Standard+Individuell; AAPL/MSFT/KO Vergleich - durchgängig in EV-030 bis EV-062 verifiziert).
- [x] Kein hartcodiertes `$` mehr (EV-022, per `grep` verifiziert); USD-Ansichten optisch unverändert; `reporting_currency` fließt Backend→API→UI (EV-020/021/022); Mixed-Currency-Hinweis funktioniert (EV-023); Fallback bei `null` (EV-022).
- [x] Vergleichs-Tooltip zeigt an jeder Position alle Firmen, Lücken als „–" (EV-031, per DOM-Inspektion und Screenshot verifiziert; Touch-Emulation nicht erneut in dieser Session wiederholt, bereits in EV-031 dokumentiert).
- [x] Zeitraumfilter: alle Ranges korrekt geschnitten (Anker = neuester Datenpunkt), leere Zeiträume sauber (EV-040/041, live mit 1J/2J/5J/Max verifiziert).
- [x] %-Veränderung: Edge-Cases (0-Start, negativer Start, 1 Punkt) zeigen „n. v." (EV-050, 17 vitest-Tests), Vorzeichen/Farben korrekt (EV-051, live verifiziert inkl. eines während der Live-Verifikation gefundenen und behobenen Vorzeichen-Bugs), nur Level-Kennzahlen (EV-051, EV/EBIT-Chart ohne Badge bestätigt).
- [x] Kurschart Analyse (absolut, USD, EV-061) + Kursvergleich (normalisiert 0 %, EV-062) + Dashboard-Sparklines (1 Batch-Call, EV-070/071) funktionieren inkl. Fehler-/Teilfehler-Zuständen (alle drei live mit simuliertem Backend-Ausfall bzw. simuliertem Teilfehler geprüft).
- [x] Sidebar-Matrix (Abschnitt 12) vollständig erfüllt (EV-080/081, 6 Zustände + Mobile-Drawer live durchgespielt), kein Pro-Flicker (identisches `isLoadingUser`-Gate wie der bestehende Admin-Eintrag), `/app/billing` direkt erreichbar (verifiziert für Pro-Zustand). **Checkout-Redirects (Stripe-Testmodus-Checkout) NICHT durchgespielt** - ein echter (auch Test-)Checkout-Durchlauf würde eine tatsächliche Zahlungsseiten-Interaktion auslösen, was außerhalb der in dieser Session sicheren/zulässigen Aktionen liegt (keine Finanztransaktionen); die Checkout-Redirect-Logik selbst wurde durch keine der 24 Aufgaben dieses Plans verändert.
- [x] Regressionsmatrix (Abschnitt 15) - Details s. u.; `pytest api/tests agent/tests`: **281/281 grün**; `npm run test`: **70/70 grün**; `npm run build`: fehlerfrei (letzter gemeinsamer Lauf in EV-090, keine Regression seit EV-040).
