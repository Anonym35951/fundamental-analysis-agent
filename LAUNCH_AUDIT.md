# LAUNCH_AUDIT.md — ComAnalysis (Final-Audit)

**Audit-Datum:** 2026-07-05 (Zweit-Audit nach Fehlerbehebungsrunde; Erst-Audit 2026-07-04)
**Prüfer:** Externe Audit-Instanz (Claude, read-only)
**Code-Stand:** `main` + uncommittete Fixes im Working Tree
**Modus:** Read-only. Einzige Schreiboperation: diese Datei.
**Ausgeführte Prüfungen (nachweislich ohne Quelldatei-Änderung):** `pytest` (28/28 grün), `npm run build` (grün, schreibt nur `dist/`), `npx eslint .` (read-only, 20 Errors / 3 Warnings), Alembic-Ketten-Analyse per Skript (read-only), bcrypt/passlib-Funktionstest in-memory.
**Nicht ausgeführt:** Migrationen, Deployments, Paketinstallationen, Schreibzugriffe auf DB/Stripe/SMTP/SEC.

> **Status-Legende:** Bewiesen (Codepfad eindeutig) · Begründetes Risiko (plausibel, an Live-Daten zu verifizieren) · Offene Frage (Produkt-/Fachentscheidung) · Subjektiv (Produkteinschätzung).
> **Prioritäten:** P0 Launch-Blocker · P1 vor Launch stark empfohlen · P2 nach Launch möglich · P3 Verbesserung.

---

## 1. Executive Summary

ComAnalysis (FastAPI-Backend + Python-Analyse-Engine + React/Vite-Frontend) hat seit dem Erst-Audit eine substanzielle Fehlerbehebungsrunde durchlaufen. **Alle sieben im Erst-Audit als kritisch/hoch markierten Code-Findings (H-1…H-3, K-1…K-4) sind im Working Tree nachweislich behoben und durch 11 neue Regressionstests abgesichert** (verifiziert in diesem Audit, siehe Abschnitt 4). Test-Suite (28/28), Production-Build und App-Import sind grün. Die Alembic-Migrationskette ist intakt (19 Revisionen, genau 1 Root, genau 1 Head, keine Lücken). Auth, Billing, Admin-Gates und Webhook-Verarbeitung sind solide gebaut.

**Verbleibende Launch-Hindernisse sind primär organisatorisch** (Impressums-/Datenschutz-Platzhalter, Produktions-Env-Werte, Stripe-Live-Keys — alles bereits in `LAUNCH_CHECKLIST.md` erfasst und weiterhin offen) **plus eine überschaubare Zahl von Code-/Fach-Punkten**: eine verbliebene Quota-Lücke (`/analyze/custom/history`), harte Analyse-Abbrüche bei legitimen Randfällen (schuldenfreie Firmen, Nicht-Dividendenzahler), die methodische Restschwäche der Wachstumsraten-Mittelung, ein fehlender In-App-Disclaimer an Kurszielen/CRV — und die weiterhin **nicht an Live-Daten verifizierte** Annahme über die Perioden-Sortierung der SEC-Daten.

**Einstufung: Soft-launch-ready mit Einschränkungen** (Begründung in Abschnitt 2).

---

## 2. Launch-Readiness-Einstufung

**Gewählte Option: 3 — Soft-launch-ready mit Einschränkungen.**

Begründung:
- **Für Launch-Fähigkeit spricht:** Der im Erst-Audit kritische Berechnungskern (Kursziele, CRV, Inflationsvergleich) ist korrigiert und testabgedeckt. Sicherheits-Grundgerüst (Auth, Autorisierung, Webhook-Signatur, Rate-Limits, CORS, Fehlerkapselung) ist verifiziert in Ordnung. Kein im Code nachweisbarer Absturz-Blocker in den zentralen Flows.
- **Gegen einen sofortigen breiten Public Launch spricht:**
  1. Die **organisatorischen P0s** (Impressum-Platzhalter = in DE nicht rechtskonform; Prod-Env/Stripe-Live) sind offen — ohne sie ist Produktion nachweislich kaputt bzw. rechtlich angreifbar.
  2. Die **fachliche Korrektheit an echten Live-Daten wurde in keinem der beiden Audits verifiziert** (kein API-Zugriff im Audit): insbesondere die Perioden-Sortierungs-Annahme (`.iloc[0]` = neueste Periode) hinter KGV/KUV/ROE/TTM und das Verhalten der 48 frisch quotierten Metrik-Endpunkte.
  3. Einzelne P1-Codepunkte (Quota-Restlücke, harte Analyse-Abbrüche bei schuldenfreien Firmen) beeinträchtigen zahlende Nutzer direkt.
- **Soft Launch** (begrenzte Nutzerzahl, aktives Monitoring via Sentry, schnelle Fix-Zyklen) ist nach Erledigung der organisatorischen P0s vertretbar. Vor einem breiten Launch sollten die P1s geschlossen und ein Live-Smoke-Test (5–10 reale Symbole × alle Analyse-Modi) durchgeführt sein.

| Bereich | Reife | Veränderung seit Erst-Audit |
|---|---|---|
| Auth / Session / Reset / Verify | 🟢 | unverändert gut |
| Billing / Stripe / Webhooks | 🟢 | unverändert gut |
| Quota / Rate-Limiting | 🟢/🟡 | H-1/H-2 behoben; eine Restlücke (P1-1) |
| Kennzahlen-Berechnung | 🟡 | K-1…K-4 behoben + Tests; Methodik-Restfragen, Live-Verifikation offen |
| Caching / Datenquellen-Last | 🟢 | H-3 behoben (TTL-Staffelung verifiziert) |
| Fehlerhandling | 🟢/🟡 | gut; harte Abbrüche in 2 Analyse-Modi (P1-3) |
| Frontend / UX | 🟢 | Build grün; Lint-Schulden, Disclaimer-Lücke |
| Tests / CI | 🟡 | 17→28 Tests; API/Auth/Billing/Frontend weiter ungetestet |
| Deployment-Bereitschaft | 🔴 organisatorisch | Env/Stripe/Impressum offen (LAUNCH_CHECKLIST.md) |

---

## 3. Wichtigste Risiken (Top 5)

1. **[Organisatorisch, P0]** Produktions-Env zeigt auf localhost/Test-Stripe; Impressum/Datenschutz enthalten Platzhalter → Launch in diesem Zustand wäre sichtbar defekt und rechtlich angreifbar.
2. **[Begründetes Risiko, P1]** Kennzahlenkern nie gegen Live-SEC-Daten verifiziert (Perioden-Sortierung, Label-Varianten realer Filer, Rate-Limit-Verhalten unter Last).
3. **[Bewiesen, P1]** `/analyze/custom/history` erlaubt unbegrenzte, quota-freie Einzelmetrik-Berechnungen — untergräbt Free-Limit und Datenquellen-Budget.
4. **[Bewiesen, P1]** Dividenden-Analyse bricht für schuldenfreie Unternehmen und Nicht-Dividendenzahler komplett ab statt das Kriterium zu bewerten — betrifft ausgerechnet die attraktivsten Dividenden-Kandidaten.
5. **[Offene Frage, P1]** Kursziel-/CRV-Ausgaben („buy_price", „Kaufzone") ohne In-App-Disclaimer — Spannungsfeld zur strikten „reine Information, keine Anlageberatung"-Positionierung.

---

## 4. Verifikation der Fixes aus dem Erst-Audit (alle bestätigt)

| ID (Erst-Audit) | Fix | Verifikation in diesem Audit |
|---|---|---|
| H-1 XFF-Spoofing | letzter statt erster XFF-Eintrag | `api/core/rate_limit.py:23` (`parts[-1]`) ✅ |
| H-2 Quota-Bypass /metrics | 48 Rechen-Endpunkte auf `require_analysis_access` | 50 Vorkommen in `metric_routes.py`; nur `current-price`/`data-source` + Router-Baseline auf `get_current_user` ✅ |
| H-3 10s-Cache-TTL | TTL-Staffelung (600d/6h/10s) | `agent/DataLoader.py:55` + `_cache_duration_for`; Klassifizierung zuvor an 34 realen Keys getestet ✅ |
| K-1 NCAV-Formel | Current Assets − Current Liabilities | `Model.py` (Course-Target-NCAV-Branch) + Test ✅ |
| K-2 TBV ohne Goodwill | `− goodwill` ergänzt | `Model.py:4254` + Test ✅ |
| K-3 stiller 0-Fallback | tolerante Label-Liste, sonst `error` | `_lookup_balance_sheet_value` + 3 Tests ✅ |
| K-4 Rate vs. kumulativ | Wachstum wird über `periods` kumuliert | `Model.py:1687, 1782` + `AgentAction.py`-Konsistenz + 3 Tests ✅ |
| Regressionstests | 11 neue Tests | `agent/tests/test_course_target_formulas.py`, 28/28 grün ✅ |

**Wichtig:** Diese Fixes liegen **uncommitted im Working Tree** (231 geänderte/neue Pfade lt. `git status`, inkl. der Cache-Bereinigung aus dem Index). Vor Launch committen — ein versehentlicher `git checkout .` würde alle Korrekturen verwerfen.

---

## 5. P0-Findings (Launch-Blocker)

### P0-1 — Impressum und Datenschutzerklärung enthalten Platzhalter
- **Kategorie:** Rechtliches · **Status:** Bewiesen
- **Dateien:** `frontend/src/pages/legal/ImprintPage.tsx:32-37` (`[Straße und Hausnummer]`, `[PLZ und Ort]`, expliziter TODO-Kommentar „Pflichtangabe nach § 5 DDG"), `frontend/src/pages/legal/PrivacyPage.tsx:38` (`[Bitte vollständige Postanschrift ergänzen]`)
- **Warum relevant:** Ohne ladungsfähige Anschrift ist das Impressum in Deutschland nicht rechtskonform; Abmahnrisiko ab Tag 1.
- **Maßnahme:** Echte Anschrift eintragen (kein Code-Problem, 10 Minuten Arbeit + Entscheidung, welche Adresse veröffentlicht wird).
- **Launch-Relevanz:** Blockierend für jeden öffentlichen Launch.

### P0-2 — Produktions-Umgebung nicht konfiguriert (Env, Stripe Live, CORS, Migrationen, Admin)
- **Kategorie:** Deployment · **Status:** Bewiesen (dokumentiert in `LAUNCH_CHECKLIST.md`, Stand geprüft und weiterhin zutreffend)
- **Beobachtung:** `FRONTEND_URL`/`STRIPE_SUCCESS_URL`/`STRIPE_CANCEL_URL` zeigen auf localhost; Stripe im Test-Modus inkl. Test-Preisen; `CORS_ORIGINS` ohne Produktions-Domain; Produktions-DB-Migrationen (`alembic upgrade head`) und Admin-Setup ausstehend; Persistent Disk für `CACHE_DIR` nötig.
- **Warum relevant:** E-Mail-Links (Verifizierung, Passwort-Reset) wären kaputt, Checkout leitet auf localhost, jeder API-Call scheitert an CORS.
- **Maßnahme:** `LAUNCH_CHECKLIST.md` Punkt für Punkt abarbeiten. Zusätzlich aus diesem Audit: **die uncommitteten Fixes committen** (siehe Abschnitt 4).
- **Launch-Relevanz:** Blockierend; rein organisatorisch.

---

## 6. P1-Findings (vor Launch stark empfohlen)

### P1-1 — `/analyze/custom/history`: quota-, verifizierungs- und ratelimit-freie Einzelmetrik-Berechnung
- **Kategorie:** Sicherheit/Abuse/Geschäftsmodell · **Status:** Bewiesen
- **Datei/Funktion:** `api/routes/custom_analysis.py:50-85` (`get_metric_history`)
- **Beobachtung:** Der Endpunkt führt via `call_metric` dieselben teuren Berechnungen aus wie die quotierten Custom-Analysis-Jobs (kompletter Metrik-Katalog, inkl. historischer Zeitreihen), hängt aber nur an `Depends(get_current_user)` — kein `require_analysis_access` (keine Quota, keine E-Mail-Verifizierungspflicht) und kein `@limiter.limit`-Dekorator.
- **Beleg:** Signatur Zeile 56-57 vs. `start_custom_analysis` Zeile 227 (`require_analysis_access_for_units`). Nach dem H-2-Fix ist dies der letzte quota-freie Rechenpfad.
- **Reproduktion:** Als Free-User (auch unverifiziert) `GET /analyze/custom/history?key=<metrik>&symbol=AAPL` in Schleife aufrufen — zählt nie gegen das Monatslimit.
- **Maßnahme:** `require_analysis_access` als Dependency + `@limiter.limit` ergänzen (identisches Muster wie der H-2-Fix in `metric_routes.py`).
- **Launch-Relevanz:** Hoch — untergräbt das Free-Limit und damit den Pro-Upgrade-Anreiz sowie das Datenquellen-Budget.

### P1-2 — Kennzahlenkern nie gegen Live-SEC-Daten verifiziert (insb. Perioden-Sortierung)
- **Kategorie:** Fachliche Korrektheit / Datenqualität · **Status:** Begründetes Risiko / Offene Frage
- **Dateien:** u. a. `agent/Model.py:70` (KGV), `:486` (KUV), `:536` (ROE) — überall `.iloc[0]` als „neueste Periode" und `.iloc[:4].sum()` als „TTM"
- **Beobachtung:** Beide Audits liefen ohne Netzwerkzugriff. Die zentrale Annahme, dass `sec_source.get_stock_financials` die Spalten absteigend nach Datum liefert, ist plausibel (Fixtures deuten darauf hin), aber nicht an frischen Live-Antworten für ein breites Symbol-Spektrum belegt. Ebenso ungeprüft: Verhalten der Label-Toleranz-Listen bei exotischeren Filern und das reale SEC-/yfinance-Rate-Limit-Verhalten der jetzt 6h-gecachten Fundamentaldaten unter Mehrnutzer-Last.
- **Maßnahme:** Vor Launch ein Live-Smoke-Test: 5–10 reale Symbole (inkl. Bank, ADR, Dienstleister ohne Inventar, schuldenfreie Firma, Nicht-Dividendenzahler) × alle 8 Analyse-Modi + ComparePage; Ergebnisse stichprobenartig gegen 10-K/10-Q-Werte prüfen. Zusätzlich Sortierung im Code explizit erzwingen (`sort_index`/`sort_values` nach Periodendatum) statt sie anzunehmen.
- **Launch-Relevanz:** Hoch — das Kernversprechen des Produkts sind korrekte Zahlen.

### P1-3 — Dividenden-Analyse bricht bei legitimen Randfällen komplett ab
- **Kategorie:** Produktlogik / Robustheit · **Status:** Bewiesen (Codepfad)
- **Dateien/Funktionen:** `agent/AgentAction.py:200-202` + `agent/Model.py:761-766`; `agent/AgentAction.py:62-63` + `agent/DataLoader.py` (`get_dividend_data`, `'dividendRate' not in stock.info` → error)
- **Beobachtung:**
  1. **Schuldenfreie Firma:** `calculate_interest_coverage_ratio` gibt bei Zinsaufwand 0 einen `error` zurück („Zinsdeckungsrate kann nicht berechnet werden") → `analyze_dividend_companies` bricht die GESAMTE Analyse mit Fehler ab (Zeile 201-202: `return {"symbol": ..., "error": ...}`). Fachlich ist „keine Zinsen" die BESTE Zinsdeckung, kein Fehlerfall.
  2. **Nicht-Dividendenzahler:** fehlt `dividendRate` in yfinance-Info → error → Komplettabbruch statt sauberem „zahlt keine Dividende → Kriterium nicht erfüllt".
  3. Gleiches Muster in `analyze_average_grower` (`fail()` bei negativem FCF, Zeile 432-433 — negativer FCF ist ein „Kriterium nicht erfüllt", kein „nicht analysierbar").
- **Kontrast:** `analyze_typical_cyclers` macht es richtig — pro Kriterium degradieren (`meets_criterion: False` + Message), Analyse läuft weiter.
- **Reproduktion:** Dividendenwerte-Analyse für ein schuldenfreies Unternehmen starten → generischer Analysefehler statt Ergebnis.
- **Maßnahme:** Teilfehler pro Kriterium behandeln (Muster aus `analyze_typical_cyclers` übernehmen); „Zinsaufwand = 0" als bestandenes Kriterium werten.
- **Launch-Relevanz:** Hoch — trifft genau die Unternehmen, für die Nutzer die Dividenden-Analyse am ehesten nutzen.

### P1-4 — AAGR/AQGR-Methodik: arithmetisches Mittel mit Verzerrungspotenzial
- **Kategorie:** Fachliche Korrektheit · **Status:** Bewiesen (Methode), Auswirkung fallabhängig
- **Dateien:** `agent/Model.py:1467-1469` (AQGR), `:1597-1599` (AAGR); Konsumenten: PEG (`:650ff`), Wachstumswerte-Kriterien, Inflationsvergleich
- **Beobachtung:** `avg_growth` ist das **arithmetische Mittel** der Perioden-Wachstumsraten. Bekannte Effekte: (a) systematischer Aufwärts-Bias ggü. geometrischem Mittel (+100 %, −50 % → Mittel +25 % bei real 0 %); (b) Basiseffekte — ein Sprung von 10 auf 500 Mio (+4900 %) dominiert den Durchschnitt; (c) AQGR vergleicht Quartal-zu-Quartal ohne Saisonbereinigung (statt YoY) — bei saisonalen Geschäften reines Rauschen; (d) übersprungene Verlustperioden verkürzen `periods`, und die K-4-Kumulierung `(1+Mittel)^n` überschätzt das echte kumulierte Wachstum zusätzlich (AM ≥ GM). Der K-4-Fix hat den Einheiten-Fehler korrigiert; die zugrunde liegende Mittelung bleibt eine Näherung.
- **Beleg:** Warnhinweis existiert bereits im eigenen Code (`AgentAction.py`: „Extrem hohes Gewinnwachstum könnte auf Datenanomalien hinweisen" ab >100 %) — das Problem ist dem Code „bekannt".
- **Maßnahme:** Auf CAGR (geometrisch, Endwert/Anfangswert) umstellen oder mindestens Median statt Mittel; AQGR auf YoY-Quartalsvergleich umstellen. PEG-Auswirkung mitprüfen.
- **Launch-Relevanz:** Hoch für Vertrauen — PEG und „Wachstumswert"-Urteile können bei volatilen Gewinnen deutlich verzerrt sein.

### P1-5 — Kein In-App-Disclaimer an Kurszielen/CRV/Kaufzonen
- **Kategorie:** UX / Rechtliches / Positionierung · **Status:** Bewiesen (Abwesenheit) + Subjektive Risikoeinschätzung
- **Dateien:** `frontend/src/components/metrics/CrvTargetPanel.tsx` (kein Treffer für Anlageberatung/Empfehlung/Hinweis-Terminologie); Disclaimer nur in `pages/legal/TermsPage.tsx` und `ImprintPage.tsx`
- **Beobachtung:** Die App zeigt „buy_price", „sell_price", „Kaufzone [1.0–1.5]", „KEIN KAUF", „ideal für Zykliker-Einstieg" (`AgentAction.py:819`) — semantisch nah an Handlungsempfehlungen. Ein kontextnaher Hinweis („keine Anlageberatung, historische Bandbreiten-Rechnung") fehlt an genau diesen Stellen; er existiert nur auf den Rechtsseiten.
- **Warum relevant:** Die dokumentierte Produkt-Leitplanke ist strikte Neutralität („reine Information, keine Anlageberatung"). Begriffe wie „KEIN KAUF" in Ergebnis-Messages stehen dazu in Spannung — sowohl rechtlich (Abgrenzung zur Anlageberatung) als auch fürs Nutzervertrauen.
- **Maßnahme:** Kurzer, permanenter Disclaimer direkt an CRV-/Kursziel-/Analyse-Panels; Wording der Ergebnis-Messages („KEIN KAUF" → „historisch Hochphase") überprüfen.
- **Launch-Relevanz:** Hoch (rechtliche Grauzone + Positionierung), Aufwand gering.

### P1-6 — Quartalsbasierte Preis-Multiples weichen von Marktkonvention ab
- **Kategorie:** Fachliche Korrektheit / UX · **Status:** Bewiesen (intern konsistent), Offene Frage (Produktentscheidung)
- **Dateien:** `agent/Model.py` (`calculate_historical_price_to_sales` u. a. nutzen Quartalsumsatz; Kursziel-Basis ebenfalls `frequency="quarterly"`)
- **Beobachtung:** P/S, P/EBIT etc. werden auf **Einzelquartals**-Basis berechnet (Preis ÷ Quartalswert/Aktie) — intern konsistent (Historie und Kursziel-Basis passen zusammen), aber die Werte sind ~4× höher als die TTM-basierte Marktkonvention. Nutzer, die mit anderen Quellen vergleichen, sehen scheinbar „falsche" Zahlen.
- **Maßnahme:** Entweder auf TTM umstellen (Historie + Kursziel-Basis gemeinsam!) oder die Quartalsbasis prominent in UI/Glossar dokumentieren. **Achtung bei Umstellung:** Beides zusammen ändern, sonst entsteht genau der Inkonsistenz-Fehlertyp, der als K-1/K-2 gerade behoben wurde.
- **Launch-Relevanz:** Mittel-hoch für Vertrauen; keine interne Rechenverfälschung.

---

## 7. P2-Findings (nach Launch möglich, aber wichtig)

| # | Finding | Status | Beleg | Maßnahme |
|---|---|---|---|---|
| P2-1 | Job-Manager hält Jobs unbegrenzt im RAM (kein Eviction) und ist Single-Instance-gebunden; Jobs gehen bei Deploy verloren (Frontend fängt 404 sauber ab) | Bewiesen | `api/services/job_manager.py` (`_jobs` wird nie geleert); Frontend-Handling `useCompare.tsx:160-166` | TTL-basierte Job-Eviction; vor Scale-Up externer Store |
| P2-2 | Full-Analysis-Job nutzt kein `make_json_safe` (datetime/np.int64 in Ergebnissen könnten Serialisierung brechen) | Begründetes Risiko | `api/routes/full_analysis.py:93` vs. `analyze.py:285` | identisch zu Single/Custom sanitizen |
| P2-3 | Quota wird verbraucht, auch wenn der Job danach an der Active-Jobs-Grenze (429) scheitert | Bewiesen | `analyze.py` (Dependency `require_analysis_access` läuft vor dem `count_active_jobs`-Check), ebenso `full_analysis.py`, `custom_analysis.py` | Active-Jobs-Check vor Quota-Verbrauch ziehen |
| P2-4 | Quartals-Negativfall zeigt Rohwerte als „Mio. USD" (Faktor 10⁶ falsch) | Bewiesen | `Model.py:1434` (`value` roh) vs. `:1563` (`value/1e6`); Message `:1438` behauptet „Mio. USD" | wie Annual-Fall durch 1e6 teilen |
| P2-5 | Dividenden-Historie/CAGR verzerrt durch angebrochenes Kalenderjahr; deprecated `resample('A')` | Bewiesen | `Model.py:240` (`YE`-Summe inkl. Teiljahr), `:310-311` (`'A'`-Alias) | laufendes Jahr ausschließen bzw. TTM; Alias aktualisieren |
| P2-6 | `print()`-Debugausgaben in Produktionspfaden; doppelte `AgentAction`-Instanz (Import-Zeit in `full_analysis.py` + Lazy-Singleton in `analyze.py`) | Bewiesen | `Model.py:545,558,568,5367,5374`; `full_analysis.py:27` | Logging statt print; eine gemeinsame Instanz |
| P2-7 | `stripe.Subscription.retrieve()` in Cancel/Resume ohne try/except → Stripe-Ausfall wird 500 statt sauberem 502 | Bewiesen | `api/routes/billing.py:122, 213` (Kontrast: `auth.py` delete-account fängt ab) | try/except + verständliche Fehlermeldung |
| P2-8 | Favorites: kein Limit pro Nutzer, keine Validierung gegen die Symbols-Tabelle → beliebige Strings landen im 6h-SEC-Filing-Worker (0,3 s Delay pro Symbol) | Bewiesen | `api/routes/favorites.py` (nur Leerstring-Check), `filing_alert_service.py` (iteriert alle distinct Symbole) | Symbol gegen `symbols`-Tabelle validieren; Obergrenze (z. B. 50) |
| P2-9 | Testlücken: keinerlei Tests für API-Routen, Auth-Flows, Quota-Logik, Stripe-Webhook, Frontend; CI führt weder Frontend-Build noch Lint aus | Bewiesen | `agent/tests/` (28 Tests, nur Engine); `.github/workflows/ci.yml` (nur pytest) | Prioritätenliste in Abschnitt 13 |
| P2-10 | ESLint: 20 Errors / 3 Warnings (setState-in-Effect-Muster, `no-explicit-any`, Fast-Refresh-Regeln); kein CI-Gate | Bewiesen | `npx eslint .`-Lauf dieses Audits (u. a. `CookieConsentBanner.tsx:23`, `useSymbolSearch.ts:18`, `MultiLayerChart.tsx:154` — letzteres Lint-Level, kein Laufzeitfehler, da Listener-Aufruf nach Deklaration) | Lint fixen + als CI-Gate |
| P2-11 | Quota-Semantik inkonsistent: Full-Analyse (13 Teil-Analysen) = 1 Einheit, Custom mit N Metriken = N Einheiten | Bewiesen / Offene Frage | `full_analysis.py:114` vs. `custom_analysis.py:227` | Produktentscheidung dokumentieren oder angleichen |
| P2-12 | `@retry`-Dekoratoren wirkungslos (Funktionen fangen intern alle Exceptions und geben error-dicts zurück → Retry feuert nie) | Bewiesen | durchgängig in `Model.py`/`DataLoader.py` | entfernen oder Retries in die HTTP-Schicht verlagern |
| P2-13 | Register-Endpunkt erlaubt Konto-Enumeration („Email already registered", „Benutzername bereits vergeben") — im Kontrast zu den enumeration-sicheren Forgot-/Resend-Flows | Bewiesen | `api/routes/auth.py:94-99` | bewusste Abwägung dokumentieren oder generische Antwort |

---

## 8. P3-Findings (Verbesserungen)

- **P3-1** Repo-Hygiene: `API Test.py` weiterhin im Git-Index (Rest erledigt: `agent/cache/*` ist raus, 0 getrackte Dateien). — `git ls-files`
- **P3-2** Tote Dependency `yahoofinance==0.0.2` (nirgends importiert; verwechslungsgefährdeter Paketname neben `yfinance`) sowie ungenutzter Import `from scipy.stats import false_discovery_control` in `AgentAction.py:5`. — Supply-Chain-Hygiene
- **P3-3** Frontend-Bundle 2,05 MB in einem Chunk (three.js + recharts); Code-Splitting würde v. a. die Landing beschleunigen. — Build-Ausgabe
- **P3-4** Text-/Code-Inkonsistenz: P/TBV-Message „Kaufzone [1.0, 1.5]" vs. Prüfung `0 < x <= 1.5` (`AgentAction.py:366-371`).
- **P3-5** Anzeige-Semantik: KGV `float('inf')` → String "inf"; negativer TBV → `tbv_per_share = 0`, P/TBV = inf statt aussagekräftiger Meldung (`Model.py:434-445`).
- **P3-6** `/full/full/{job_id}/…`-Doppelpfad (Router-Prefix + Routen-Pfad) — funktioniert, verwirrt aber (`full_analysis.py:22,137`).
- **P3-7** `datetime.utcnow()` durchgängig (deprecated, naive) — konsistent, aber mittelfristig auf tz-aware migrieren.

---

## 9. Fachliche Prüfung der Kennzahlen (Zusammenfassung)

**Verifiziert korrekt (Formel-Ebene, gegen Code + Tests):** EBITDA = EBIT + D&A (inkl. Bank-Sonderfall JPM), FCF = OCF − CapEx, historischer TBV (inkl. Goodwill), Kursziel-Formeln nach K-1/K-2/K-3-Fix (per Regressionstest mit Wrong-Value-Guards), Inflationsvergleich nach K-4-Fix (kumuliert vs. kumuliert), Dividendenrendite-Einheiten (Prozent, `DataLoader.get_dividend_data`), ROE-/CFM-/Inventory-Einheiten in Zykliker-Analyse, Buy/Sell/FV/WC-Szenariologik (Median aus 3 Jahres-Extremen, WC = Buy/1.2 — nachvollziehbare, dokumentierte Heuristik).

**Offen/risikobehaftet:** P1-2 (Perioden-Sortierung live), P1-4 (Mittelungs-Methodik), P1-6 (Quartals- vs. TTM-Basis), P2-4/P2-5 (Anzeige-/Teiljahr-Verzerrungen). CRV-Downside-Konvention „Kurs < Worst-Case → Downside = voller Kurs" (`Model.py`, `calculate_crv`) ist sehr konservativ — vertretbar, sollte aber im Glossar erklärt werden (Offene Frage).

**Vergleichbarkeit (ComparePage):** identische Metrik-Dispatch-Logik pro Unternehmen (gleicher Katalog, gleiche Frequenz erzwungen durch Coverage-Check in `useCompare.tsx`) — Vergleiche sind konsistent, sofern die Einzelmetriken stimmen. Kein Scoring/Ranking implementiert, daher kein Ranking-Verzerrungsrisiko.

---

## 10. Nutzerflüsse (geprüft auf Code-Ebene)

- **Registrierung → Verifizierung → Login:** vollständig; Login blockiert unverifizierte Konten mit klarer Meldung + öffentlichem Resend-Weg (`/auth/resend-verification-public`, enumeration-sicher). ✅
- **Passwort vergessen/ändern:** Token-Hashing, TTL, Bestätigungsmail, generische Antworten. ✅
- **Analyse (Job-basiert):** Start → Progress-Polling → Ergebnis; Job-Verlust nach Deploy wird im Frontend erklärt; Quota-Überschreitung hat eigenes Modal (`QuotaExceededModal.tsx`) mit Reset-Datum. ✅
- **Vergleich:** Workspace persistiert, wird bei Logout über `app:logout`-Event zuverlässig geleert (`client.ts:47`, `useCompare.tsx`). ✅
- **Billing:** Checkout → Webhook → Plan; Kündigen/Fortsetzen inkl. Mails; Grace-Period; Konto-Löschung kündigt Abo zuerst und bricht bei Stripe-Fehler ab. ✅
- **Admin:** alle Routen serverseitig `require_admin`-gegated; Frontend-Guard nur UX. ✅
- **Schwachstellen in Flows:** P1-3 (Dividenden-Randfälle), P2-3 (Quota bei 429), P1-1 (History-Endpunkt).

## 11. Sicherheit & Datenschutz (geprüft)

Verifiziert in Ordnung: JWT-Handling (401-Pfad, aber siehe unten), bcrypt 4.0.1 + passlib funktionsfähig (in-memory getestet), Webhook-Signatur + Idempotenz, CORS restriktiv ohne Credentials, Rate-Limits auf allen Auth-Endpunkten (nach H-1-Fix nicht mehr spoofbar), Job-Isolation pro User, SQL ausschließlich über ORM (keine Roh-SQL-Injection-Fläche gefunden), E-Mail-Bau mit `html.escape` + `EmailStr` (keine Header-Injection), keine Secrets im Git-Index (`.env` untracked; Werte hier bewusst nicht wiedergegeben), Fehlermeldungen generisch ohne interne Details.

Restpunkte: Token in `localStorage` (XSS-Tradeoff, bewusst; ohne Cookie-CSRF-Fläche konsistent), keine Token-Revocation (30-min-Expiry begrenzt das Fenster), P2-13 (Register-Enumeration), P1-1 (ungedrosselter Rechen-Endpunkt), P2-8 (Favorites-Abuse). CSRF: nicht relevant (kein Cookie-Auth). Offene Redirects: keine gefunden (Redirect-Ziele stammen aus Settings, nicht aus Request-Parametern).

## 12. Performance & Skalierung (geprüft)

Nach H-3 ist die Datenquellen-Last strukturell entschärft (6h-TTL Fundamentaldaten, verifizierte Klassifizierung). Verbleibend: alle Analysen laufen synchron in einem 4-Worker-Threadpool (`job_manager`), yfinance `.info`-Calls (Preis, Profil, Dividenden) sind der langsamste Pfad; ComparePage startet 1 Job pro Firma (parallel, gedeckelt auf 3 aktive Jobs/User). Symbol-Suche ist serverseitig mit `ilike` auf ~7k Zeilen + Rate-Limit — unkritisch. Frontend: 2-MB-Bundle (P3-3), Polling 1,5 s/Job und 20 s/Preis mit In-Process-Caches — vertretbar. Single-Instance-Constraint ist dokumentiert (`LAUNCH_CHECKLIST.md`) und für den Start akzeptabel.

## 13. Tests & QA (Bewertung)

**Vorhanden:** 28 Engine-Tests (Formel-Identitäten gegen eingefrorene echte SEC-Fixtures, ein gegen das reale 10-Q verifizierter Wert, Sonderfälle Bank/ADR, 11 Kursziel-/Inflations-Regressionstests mit Wrong-Value-Guards). Qualität der vorhandenen Tests: gut, nicht tautologisch.
**Fehlt (Priorität):** (1) API-Tests Auth-Flows (Login/Verify/Reset inkl. Fehlerpfade), (2) Quota-Atomik + on-read-Reset + 429-Pfad, (3) Stripe-Webhook-Handler (Status-Übergänge, Idempotenz) mit gemockten Events, (4) `/analyze/custom/history`-Zugriffsregeln (nach P1-1-Fix), (5) Frontend: `useCompare`-Staleness, 401-Redirect, QuotaModal, (6) E2E-Smoke (Register→Verify→Analyse→Ergebnis). CI sollte zusätzlich Frontend-Build + ESLint fahren (P2-9/P2-10).

## 14. UX & Vertrauen (Bewertung)

Stark: Quellen-Badge + Datenstand (`/metrics/data-source`), Datenquellen-Statusseite, Glossar (94 Seiten, SEO-generiert), Quota-Modal mit Reset-Datum, klare Job-Verlust-Meldung, Cookie-Consent, Onboarding-Tour. Schwächen: P1-5 (Disclaimer an Kurszielen), P1-6 (Multiple-Konvention unerklärt), P2-4 (falsche Mio-Angabe), imperative Formulierungen („KEIN KAUF") in Ergebnis-Messages. Mobile/responsive wurde nur statisch geprüft (Media-Query-Hooks vorhanden) — keine Aussage über tatsächliches Verhalten auf Geräten.

## 15. Deployment & Produktion (Bewertung)

Alembic-Kette verifiziert sauber (19 Revisionen, 1 Root → 1 Head, keine fehlenden `down_revision`). Dependencies vollständig gepinnt; bcrypt-Version kompatibel. Sentry-Anbindung code-seitig fertig (aktiviert sich per Env). Worker-Architektur (Downgrade/Filing/Digest im Webprozess) dokumentiert inkl. Betriebsauflagen. Offen: alles aus P0-2 sowie `robots.txt`/`sitemap.xml` (Sitemap wird für den Glossar generiert — `vercel.json` rewritet sie korrekt; robots.txt nicht gefunden, P3).

---

## 16. Geprüfte Bereiche (dieses Final-Audit)

Fix-Verifikation (H-1…H-3, K-1…K-4, Tests); `agent/AgentAction.py` vollständig (alle 8 Analyse-Modi); `agent/DataPreprocessor.py` (Kopf + Indikatoren); `agent/data_sources/sec_source.py` (HTTP-Schicht: Timeouts, UA, companyfacts-Quelle); `api/routes/favorites.py`, `status.py`, `admin_stats.py`, `admin_customers.py` (Gates); `api/services/`: `custom_analysis_limit`, `event_service`, `stripe_event_service`, `filing_alert_service`, `data_source_status_service`, `metric_catalog` (Kopf/Dispatch); `api/crud/favorite.py`; Frontend: Legal-Seiten, Quota-UX, Disclaimer-Suche, `MultiLayerChart`-Lint-Befund; Alembic-Kette (skriptbasiert); `requirements.txt` + bcrypt-Funktionstest; ESLint-Gesamtlauf; pytest; Production-Build. Plus alle im Erst-Audit gelisteten Bereiche (siehe Historie unten).

## 17. Nicht geprüfte Bereiche (und warum)

- **Live-Verhalten SEC/yfinance/Alpha Vantage/FRED** — kein Netzwerk-/API-Zugriff im Audit-Modus; deshalb P1-2 als explizite Auflage.
- **Stripe End-to-End (echter Checkout/Webhook)** — keine produktiven Schreibzugriffe erlaubt.
- **E-Mail-Zustellung (SMTP)** — kein Versand im Audit.
- **`agent/Model.py` restliche ~15 %** (einzelne historische Multiple-Varianten, Elliott-Wave-Experimentalcode) — Priorisierung auf launch-relevante Pfade.
- **`agent/data_sources/sec_source.py` Parsing-Kern (~2.800 Zeilen XBRL-Tag-Mapping)** — nur HTTP-/Cache-Schicht geprüft; das Tag-Mapping ist durch die Fixture-Tests indirekt, aber nicht vollständig abgedeckt.
- **Tatsächliches Mobile-/Browser-Verhalten** — kein Browser-Lauf im Audit (statische Prüfung der Responsive-Hooks).
- **Migrationen inhaltlich pro Revision** — nur Kettenintegrität verifiziert.
- **Lasttest/Concurrency real** — nur architektonische Bewertung.

## 18. To-do-Liste vor Launch (dokumentiert, NICHT umgesetzt)

**Vor jedem Launch (P0):**
1. Impressum + Datenschutz: echte Anschrift (`ImprintPage.tsx`, `PrivacyPage.tsx`).
2. `LAUNCH_CHECKLIST.md` abarbeiten (Env, Stripe Live, CORS, Migrationen, Admin, Persistent Disk, Sentry).
3. **Uncommittete Fixes committen** (Working Tree enthält alle Korrekturen aus der Fehlerbehebungsrunde).

**Vor Public Launch (P1):**
4. `/analyze/custom/history` quotieren + ratelimiten (P1-1).
5. Live-Smoke-Test 5–10 Symbole × 8 Modi; Perioden-Sortierung explizit erzwingen (P1-2).
6. Dividenden-/Average-Grower-Randfälle degradieren statt abbrechen (P1-3).
7. In-App-Disclaimer an Kursziel-/CRV-Panels; „KEIN KAUF"-Wording entschärfen (P1-5).
8. Entscheidung AAGR→CAGR/Median + AQGR→YoY (P1-4) und TTM-vs-Quartal (P1-6) — Methodik-Festlegung dokumentieren.

**Zeitnah nach Launch (P2):** Punkte P2-1 bis P2-13 in der Reihenfolge der Tabelle; Testausbau gemäß Abschnitt 13.

## 19. Offene Fragen (Produkt-/Fachentscheidungen)

1. Quota-Semantik: Full = 1 Einheit vs. Custom = N Einheiten — gewollt? (P2-11)
2. Multiple-Basis: Quartal beibehalten (und erklären) oder TTM? (P1-6)
3. CRV-Downside-Konvention (Kurs < WC → Totalverlust-Annahme) — beibehalten und im Glossar erklären?
4. Register-Enumeration bewusst in Kauf nehmen (UX) oder generifizieren? (P2-13)
5. Favorites-Obergrenze und Symbol-Validierung — Produktlimit definieren. (P2-8)

## 20. Finale Empfehlung

**Soft-launch-ready mit Einschränkungen.** Nach Erledigung der organisatorischen P0s (Impressum, Prod-Env/Stripe, Commit der Fixes) ist ein kontrollierter Soft Launch mit begrenzter Nutzerzahl und aktivem Sentry-Monitoring vertretbar — die zentralen Flows sind code-seitig robust und der Berechnungskern ist nach der Fixrunde deutlich belastbarer und testgesichert. **Ein breiter Public Launch sollte erst nach** dem Live-Smoke-Test (P1-2), dem Schließen der Quota-Restlücke (P1-1), den Analyse-Randfällen (P1-3) und dem In-App-Disclaimer (P1-5) erfolgen; die Methodik-Entscheidungen (P1-4, P1-6) sollten davor bewusst getroffen und dokumentiert sein — sie bestimmen, ob Nutzer den Kennzahlen dauerhaft vertrauen.

---

## Anhang: Historie

- **2026-07-04 (Erst-Audit):** 7 kritische/hohe Findings (H-1…H-3, K-1…K-4) + Mittel-/Niedrig-Findings; Einstufung „nicht marktreif".
- **2026-07-04/05 (Fixrunde):** alle 7 Findings behoben, 11 Regressionstests ergänzt (28/28 grün).
- **2026-07-05 (dieses Final-Audit):** Fixes verifiziert; verbleibende Findings neu priorisiert (P0: 2 organisatorisch · P1: 6 · P2: 13 · P3: 7); Einstufung angehoben auf „Soft-launch-ready mit Einschränkungen".
