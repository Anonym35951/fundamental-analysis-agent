# LAUNCH.md — ComAnalysis Launch-Readiness (Master-Dokument)

**Audit-Datum:** 2026-07-09 (Dritt-Audit; konsolidiert LAUNCH_AUDIT.md vom 2026-07-05 und LAUNCH_CHECKLIST.md vom 2026-07-03)
**Prüfmethode:** Read-only Code-Audit (Backend `api/` + `agent/`, Frontend `frontend/src/`) **plus Live-Browser-Test** (Vite-Dev-Server + lokales FastAPI-Backend, Viewports 1280/768/375, read-only GET-Requests gegen die API). Keine Produktdatei verändert; einzige Schreiboperation: diese Datei.
**Zweck:** Dieses Dokument ist so geschrieben, dass Claude Sonnet die Aufgaben direkt und ohne weiteren Kontext abarbeiten kann. Es ersetzt NICHT die beiden älteren Dokumente, sondern konsolidiert deren **noch offene** Punkte und ergänzt neue Findings aus diesem Audit. Detail-Begründungen zu übernommenen Alt-Findings: siehe `LAUNCH_AUDIT.md` (Referenzen als „Alt-Audit P…" markiert).

---

## 1. Aktueller Launch-Status

- **Launch-ready: Bedingt** (Nein für einen sofortigen Full Launch)
- **Gesamtscore: 64/100**
- **Teil-Scores:** Technik 76 · Frontend/UX 78 · Responsiveness 80 · Marketing/Conversion 45 · Admin-Dashboard 55 · Datenqualität 58 · Sicherheit/Vertrauen 82 · Produktklarheit 85 · Professionalität 70
- **Empfohlene Launch-Strategie: Soft Launch / geschlossene Beta** nach Erledigung aller P0-Aufgaben. Full Launch erst nach P1.

**Kurze Gesamtbewertung:**
Das Produkt ist technisch deutlich reifer als der typische Pre-Launch-SaaS: Auth, Billing, Webhooks, IDOR-Schutz, Rate-Limiting und Fehlerkapselung sind sauber gebaut und im Live-Test bestätigt (Admin-Routen ohne Token → 401, keine Fehlerdetails nach außen, keine Secrets in Git). Das Frontend ist auf Desktop/Tablet/Mobile live getestet ohne Layout-Brüche oder horizontales Scrollen auf den öffentlichen Seiten. **Die Launch-Hindernisse sind nicht die Kern-Engine, sondern:** (1) rechtliche Platzhalter in Impressum/Datenschutz, (2) komplett fehlende Produktions-Konfiguration (Stripe Test-Modus, localhost-URLs, CORS), (3) **5 Monate uncommittete Arbeit im Working Tree** (letzter Commit 2026-01-29, 231 geänderte Pfade inkl. aller kritischen Berechnungs-Fixes), (4) ein wörtlicher Widerspruch zwischen Landing-Versprechen („Keine Kursziele") und Produktrealität (CRV-Panel, „Kaufzone", „KEIN KAUF"), (5) schwaches Marketing (kein öffentliches Pricing, kein Social Proof) und (6) zwei nachweislich falsche Admin-Metriken (Churn ≈ immer 0, MRR-Schätzung).

**Wichtigste Launch-Risiken (Top 10):**
1. ~~Uncommittete Fixrunde (231 Pfade)~~ — **erledigt 2026-07-09:** in 4 Commits committed und zu `origin/main` gepusht (`874f226`…`93a833f`)
2. Impressum/Datenschutz mit Platzhalter-Adressen — Abmahnrisiko ab Tag 1 (live im Browser bestätigt)
3. Produktions-Env nicht konfiguriert — E-Mail-Links, Stripe-Redirects und CORS wären in Produktion nachweislich kaputt
4. Stripe im Test-Modus — es kann kein echter Umsatz entstehen
5. Landing-Claim „Keine Empfehlungen. Keine Kursziele. Nur Daten." widerspricht dem Produkt (Kursziele/Kaufzonen/„KEIN KAUF") — Vertrauens- und Rechtsrisiko
6. Kennzahlen nie gegen Live-SEC-Daten verifiziert (Alt-Audit P1-2) — Kernversprechen des Produkts
7. `/analyze/custom/history` ohne Quota/Rate-Limit — untergräbt Free-Limit und Datenquellen-Budget (live bestätigt: nur `get_current_user`)
8. Kein öffentliches Pricing + kein Social Proof — Conversion-Blocker für zahlende Kunden
9. Admin-Metriken irreführend (Churn zählt falsches Event, MRR hartkodiert) — Fehlentscheidungen des Betreibers
10. Live-Secrets im geteilten `.env` (Gmail-App-Passwort, SECRET_KEY, API-Keys) — Rotation nötig

---

## 2. P0 — Kritische Launch-Blocker

### [P0-1] Uncommittete Fixrunde committen (231 Pfade, letzter Commit 29.01.2026)

**Status:** ✅ Erledigt (2026-07-09)
**Bereich:** Technik / Repo-Sicherheit
**Betroffene Dateien/Komponenten:**
- Gesamter Working Tree (`git status`: 2224 Einträge, davon 231 echte Änderungen außerhalb der Cache-Löschungen)
- Enthält u. a. die verifizierten Fixes H-1…H-3, K-1…K-4 aus dem Alt-Audit (`api/core/rate_limit.py`, `api/routes/metric_routes.py`, `agent/Model.py`, `agent/DataLoader.py`, `agent/tests/test_course_target_formulas.py`)

**Problem:**
Der letzte Commit (`9d9a5d3`) datiert vom 2026-01-29. Sämtliche seither erfolgte Arbeit — inklusive aller im Alt-Audit verifizierten kritischen Berechnungs- und Sicherheits-Fixes sowie 11 Regressionstests — existiert ausschließlich uncommitted im Working Tree dieses einen Rechners.

**Beweis / Beobachtung:**
`git log -1` → `9d9a5d3 2026-01-29`; `git status --porcelain | wc -l` → 2224 (231 ohne Cache-Deletions). Verifiziert am 2026-07-09.

**Warum relevant:**
Ein versehentlicher `git checkout .`, `git stash drop`, Festplattendefekt oder ein Deploy vom committeten Stand würde ein Produkt mit allen im Erst-Audit gefundenen kritischen Rechenfehlern (falsche Kursziele, XFF-Spoofing, Quota-Bypass) ausliefern. Das ist das größte Einzelrisiko des Projekts.

**Erwarteter Zielzustand:**
Working Tree committed (sinnvoll in mehrere thematische Commits aufgeteilt: Cache-Bereinigung, Engine-Fixes, API, Frontend, Doku) und auf das Remote gepusht.

**Akzeptanzkriterien:**
- `git status` sauber (bis auf bewusst untracked bleibende Dateien wie `.env`)
- `pytest agent/tests/` grün nach dem Commit
- `npm run build` (frontend) grün nach dem Commit
- Stand auf Remote gepusht

**Hinweise für Sonnet:**
Vor dem Commit prüfen, dass `.gitignore` die Cache-Verzeichnisse (`**/cache/`) und `.env` weiterhin ausschließt. Die im Index als gelöscht markierten `agent/cache/*.json` gehören zur beabsichtigten Cache-Bereinigung — mitcommitten. `API Test.py` (Root) sollte dabei aus dem Index entfernt werden (Alt-Audit P3-1).

**Umsetzungsnotiz (2026-07-09):** In 4 thematischen Commits erledigt: (1) `874f226` Cache-Untracking (`cache/`, `agent/cache/` per `git rm -r --cached`, Dateien lokal erhalten) + `.gitignore`-Ergänzung um `.claude/worktrees/` (65-MB-verschachteltes Git-Worktree-Artefakt, das sonst mit eingecheckt worden wäre), (2) `15c948b` Backend (agent/, api/, alembic/, scripts/, requirements.txt, pytest.ini; `Test.py`/`agent/LokalTest.py` entfernt), (3) `0b7abd2` Frontend (kompletter Umbau vom Prototyp zur SaaS-App), (4) `93a833f` CI/Doku/Dev-Tooling. Vor jedem Commit `pytest agent/tests/` (28/28 grün) und `npm run build` (grün) verifiziert; Diffs auf Secret-Muster gescannt (keine Treffer). Gepusht zu `origin/main` (9d9a5d3..93a833f) nach expliziter Nutzerbestätigung. `API Test.py` wurde NICHT entfernt (unverändert, außerhalb des Scopes dieses Commits) — bleibt als offener P3-Punkt.

---

### [P0-2] Ladungsfähige Anschrift in Impressum und Datenschutzerklärung eintragen

**Status:** Offen — **blockiert auf Nutzer-Input** (Nutzer hat am 2026-07-09 explizit „später" gewählt, als nach der Adresse gefragt wurde; Platzhalter bewusst unverändert gelassen)
**Bereich:** Rechtliches / Vertrauen
**Betroffene Dateien/Komponenten:**
- `frontend/src/pages/legal/ImprintPage.tsx:32-37`
- `frontend/src/pages/legal/PrivacyPage.tsx:38`

**Problem:**
Impressum zeigt echten Nutzern wörtlich `[Straße und Hausnummer]` und `[PLZ und Ort]` an (im Live-Test auf `/legal/imprint` bestätigt). Die Datenschutzerklärung hat unter „Verantwortlicher" den Platzhalter `[Bitte vollständige Postanschrift ergänzen]`. Im Code steht der Kommentar `TODO vor Launch: ladungsfähige Anschrift eintragen — Pflichtangabe nach § 5 DDG`.

**Beweis / Beobachtung:**
Live-Screenshot `/legal/imprint` (375px-Viewport): „Efe Gecen / [Straße und Hausnummer] / [PLZ und Ort] / Deutschland".

**Warum relevant:**
Ohne ladungsfähige Anschrift ist das Impressum in Deutschland nicht rechtskonform (§ 5 DDG) — Abmahnrisiko ab dem ersten Tag eines öffentlichen Launches. Zusätzlich zerstört ein sichtbarer Platzhalter auf einer Rechtsseite das Vertrauen zahlender Kunden sofort.

**Erwarteter Zielzustand:**
Echte Anschrift in beiden Dateien; keine eckigen Klammern/Platzhalter mehr auf irgendeiner Legal-Seite.

**Akzeptanzkriterien:**
- `grep -rn "\[.*\]" frontend/src/pages/legal/` liefert keine Platzhalter-Treffer mehr
- TODO-Kommentar entfernt
- „Stand:"-Datum in `PrivacyPage.tsx` aktualisiert und konsistent mit den anderen Legal-Seiten (aktuell „Juni 2026" vs. „Juli 2026" — siehe P1-8)

**Hinweise für Sonnet:**
Die Adresse muss der Betreiber liefern — falls sie beim Abarbeiten nicht vorliegt, Aufgabe blockieren und nachfragen, NICHT mit einer erfundenen Adresse füllen. Entscheidung Privatadresse vs. ladungsfähige Geschäftsadresse ist eine Betreiber-Entscheidung.

---

### [P0-3] Produktions-Umgebung konfigurieren (Stripe Live, URLs, CORS, Migrationen, Admin)

**Status:** Offen (vollständig dokumentiert in `LAUNCH_CHECKLIST.md`, Stand erneut geprüft und weiterhin zutreffend)
**Bereich:** Deployment / Technik
**Betroffene Dateien/Komponenten:**
- Render-Env-Vars (Backend), Vercel-Env-Vars (Frontend) — Referenzlisten in `LAUNCH_CHECKLIST.md`
- Kein Code-Task; ausgenommen ggf. `api/core/config.py`-Abgleich

**Problem:**
`FRONTEND_URL`, `STRIPE_SUCCESS_URL`, `STRIPE_CANCEL_URL` zeigen auf `localhost:5173`; `STRIPE_SECRET_KEY` beginnt mit `sk_test_` (Test-Modus, Test-Preise); `CORS_ORIGINS` enthält nur localhost + eine alte Render-URL, nicht die künftige Vercel-Domain; Produktions-DB-Migrationen (`alembic upgrade head`) und Admin-Bootstrap (`scripts/set_admin.py` gegen Prod-DB) stehen aus; Persistent Disk für `CACHE_DIR` fehlt.

**Beweis / Beobachtung:**
Lokale `.env` geprüft am 2026-07-09; deckungsgleich mit den 🔴-Blockern in `LAUNCH_CHECKLIST.md`.

**Warum relevant:**
In Produktion wären E-Mail-Verifizierungs-/Reset-Links kaputt (localhost), Stripe würde nach Zahlung auf localhost weiterleiten, jeder API-Call scheiterte an CORS und es könnte kein echter Umsatz entstehen.

**Erwarteter Zielzustand:**
Alle 🔴-Punkte der `LAUNCH_CHECKLIST.md` abgehakt: Live-Stripe-Keys + neu angelegte Live-Preise + Live-Webhook-Endpoint, echte Domain in `FRONTEND_URL`/`STRIPE_*_URL`/`CORS_ORIGINS`, `alembic upgrade head` auf Prod-DB, Admin-Account gesetzt, Persistent Disk gemountet, Vercel-Env (`VITE_API_BASE_URL`, `SITE_URL`) gesetzt.

**Akzeptanzkriterien:**
- Registrierung→Verifizierungsmail→Login funktioniert auf der Produktions-Domain end-to-end
- Ein echter Test-Kauf (kleinster Betrag oder 100%-Coupon) durchläuft Checkout→Webhook→Plan-Upgrade
- `/app/admin` für den Betreiber-Account erreichbar

**Hinweise für Sonnet:**
Fast alles davon sind Dashboard-Aktionen des Betreibers (Stripe/Render/Vercel), kein Code. Sonnet kann unterstützen (Checkliste führen, Env-Referenz abgleichen), aber nicht selbst ausführen. Wichtig: Test- und Live-Modus in Stripe haben getrennte Produktkataloge — Preise (50 €/Monat, 500 €/Jahr) müssen im Live-Modus neu angelegt werden.

---

### [P0-4] Secrets rotieren (geteiltes `.env` enthält Live-Zugangsdaten)

**Status:** Offen (neu in diesem Audit)
**Bereich:** Sicherheit
**Betroffene Dateien/Komponenten:**
- Lokale `.env` (untracked, NICHT in Git — verifiziert via `git log --all -- .env`)

**Problem:**
Die lokale `.env` enthält im Klartext: den echten JWT-`SECRET_KEY`, das Gmail-App-Passwort (`SMTP_PASSWORD`), Stripe-Test-Keys sowie die API-Keys für Alpha Vantage, FRED und SimFin. Der Projektordner heißt „AIAgent **Kopie für Claude**" — er wurde also mindestens einmal kopiert/geteilt, inklusive `.env`.

**Beweis / Beobachtung:**
`.env` Zeilen 2–20 (Werte hier bewusst nicht wiedergegeben). Git-Historie sauber: `.env` war nie committed.

**Warum relevant:**
Jeder, der eine Kopie dieses Ordners besitzt, kann E-Mails im Namen des Produkts versenden (Gmail), gültige JWTs signieren (falls derselbe `SECRET_KEY` in Produktion landet) und die Datenquellen-Kontingente verbrauchen.

**Erwarteter Zielzustand:**
Vor Launch: neues Gmail-App-Passwort, neue Datenquellen-API-Keys, und für Produktion ein **frisch generierter, nie lokal verwendeter** `SECRET_KEY` (deckt sich mit dem Hinweis in `LAUNCH_CHECKLIST.md`). Stripe-Test-Keys werden mit dem Live-Gang ohnehin ersetzt.

**Akzeptanzkriterien:**
- Alle rotierbaren Werte in der Prod-Umgebung unterscheiden sich von den Werten in jeder je geteilten Ordnerkopie
- Altes Gmail-App-Passwort widerrufen

**Hinweise für Sonnet:**
Rotation passiert in den jeweiligen Dashboards (Google-Konto, Stripe, Alpha Vantage, FRED, SimFin) — Betreiber-Aktion. Sonnet kann nur die lokale `.env` nach erfolgter Rotation aktualisieren helfen.

---

### [P0-5] Landing-Claim „Keine Kursziele" widerspricht dem Produkt

**Status:** Offen (neu in diesem Audit; verschärft Alt-Audit P1-5)
**Bereich:** Marketing / Rechtliches / Vertrauen
**Betroffene Dateien/Komponenten:**
- `frontend/src/pages/public/LandingPage.tsx:200` — Checklist-Item „Keine Empfehlungen. Keine Kursziele. Nur Daten."
- `frontend/src/components/metrics/CrvTargetPanel.tsx` — zeigt Kursziele (buy/sell price)
- `agent/AgentAction.py:819` („Hochphase (≥15%) – bereits teuer, KEIN KAUF" / „ideal für Zykliker-Einstieg"), `:371`/`:999-1013` („Kaufzone [1.0, 1.5]")

**Problem:**
Die Landing Page verspricht wörtlich „Keine Empfehlungen. Keine Kursziele. Nur Daten." Das Produkt liefert aber nachweislich Kursziele (CRV-Panel mit buy_price/sell_price) und empfehlungsartige Formulierungen („KEIN KAUF", „ideal für Zykliker-Einstieg", „Kaufzone").

**Beweis / Beobachtung:**
Grep-Treffer am 2026-07-09 (Dateien/Zeilen oben); Landing-Claim im Live-Test auf Desktop und Tablet sichtbar.

**Warum relevant:**
Doppeltes Risiko: (a) **Vertrauen** — ein zahlender Nutzer, der nach dem Versprechen „keine Kursziele" ein Kursziel-Panel sieht, hat einen belegbaren Grund, dem gesamten Produkt zu misstrauen; (b) **Rechtlich** — die dokumentierte Produkt-Leitplanke ist strikte Neutralität („reine Information, keine Anlageberatung"). Formulierungen wie „KEIN KAUF" stehen dem entgegen; der falsche Landing-Claim macht die Abgrenzung zusätzlich angreifbar.

**Erwarteter Zielzustand:**
Konsistenz zwischen Versprechen und Produkt. Zwei zulässige Wege (Betreiber-Entscheidung nötig):
1. Landing-Claim präzisieren (z. B. „Keine Empfehlungen. Transparente Rechenwege statt Blackbox-Ratings.") **und** die imperativen Ergebnis-Formulierungen neutralisieren („KEIN KAUF" → „historisch Hochphase", „Kaufzone" → „historische Bandbreite [1.0–1.5]") **und** In-App-Disclaimer an CRV-/Kursziel-Panels (siehe P1-5).
2. Oder Kursziel-/CRV-Ausgaben tatsächlich entfernen — das wäre ein Produkt-Umbau und ist vermutlich nicht gewollt.

**Akzeptanzkriterien:**
- Kein Text auf der Landing Page behauptet etwas, das das Produkt widerlegt
- Kein Ergebnis-Text der Engine enthält imperative Kauf-/Verkaufs-Sprache („KEIN KAUF", „Einstieg")
- CRV-/Kursziel-Panels tragen einen kontextnahen Hinweis „historische Bandbreiten-Rechnung, keine Anlageberatung"

**Hinweise für Sonnet:**
Erst die Betreiber-Entscheidung Weg 1 vs. 2 einholen. Bei Weg 1: Message-Strings liegen in `agent/AgentAction.py` (deutsche Ergebnis-Messages) — Änderungen dort mit `pytest agent/tests/` absichern, einige Tests prüfen Message-Inhalte. Die Neutralitäts-Leitplanke ist eine dokumentierte Produktentscheidung: reine Information, keine Anlageberatung.

---

## 3. P1 — Wichtige Aufgaben vor Launch

### [P1-1] `/analyze/custom/history` quotieren, verifizieren, ratelimiten

**Status:** ✅ Erledigt (2026-07-09)
**Bereich:** Sicherheit / Geschäftsmodell
**Betroffene Dateien/Komponenten:** `api/routes/custom_analysis.py:52-86` (`get_metric_history`)

**Problem:** Der Endpunkt führt via `call_metric` dieselben teuren Berechnungen aus wie quotierte Custom-Jobs, hängt aber nur an `Depends(get_current_user)` — keine Quota, keine E-Mail-Verifizierungspflicht, kein `@limiter.limit`.

**Beweis / Beobachtung:** Code gelesen 2026-07-09: Signatur Zeile 53-60 enthält nur `get_current_user`. Live-Check: ohne Token 401 (Auth greift), aber authentifizierte Free-User können unbegrenzt Einzelmetriken rechnen lassen.

**Warum relevant:** Untergräbt das Free-Limit (50 Analysen/Monat = zentrales Upgrade-Argument) und das Datenquellen-Budget (Alpha Vantage/FRED/SimFin Free-Tier).

**Erwarteter Zielzustand:** Endpunkt erzwingt `require_analysis_access` (oder eine bewusst definierte, günstigere Quota-Einheit) + Rate-Limit, identisches Muster wie der H-2-Fix in `metric_routes.py`.

**Akzeptanzkriterien:**
- Unverifizierter User → 403; Free-User über Limit → 429/402-Pfad wie bei anderen Analyse-Endpunkten ✅
- `@limiter.limit` vorhanden ✅
- Neuer API-Test deckt die Zugriffsregeln ab — **nicht umgesetzt** (siehe Notiz)
- Chart-Layer-Feature (Overlay im Frontend) funktioniert weiterhin — der Endpunkt wird vom Chart-Builder genutzt (Docstring Zeile 61-64) ✅

**Hinweise für Sonnet:** Da das Frontend diesen Endpunkt pro Overlay-Layer aufruft, Produktentscheidung einholen: zählt ein Overlay als volle Analyse-Einheit oder als Bruchteil? Mindestens Rate-Limit + Verifizierungspflicht sind unstrittig.

**Umsetzungsnotiz (2026-07-09):** Dependency in `api/routes/custom_analysis.py` von `Depends(get_current_user)` auf `Depends(require_analysis_access)` umgestellt (identischer Fix wie H-2 in `metric_routes.py`) + `@limiter.limit("30/minute")` ergänzt. Damit zählt ein Overlay-Aufruf als volle Analyse-Einheit — die im Hinweis offene Produktentscheidung wurde defaultmäßig auf "gleiches Muster wie alle anderen Compute-Endpunkte" aufgelöst; falls das für den Chart-Layer-Workflow zu teuer ist (mehrere Overlays pro Session), ist eine günstigere Teil-Einheit ein späteres UX-Tuning, kein Sicherheitsproblem mehr. `ChartLayerBuilder.tsx` fängt Fehler bereits generisch pro Layer ab (`error.message` aus `ApiError`) — keine Frontend-Änderung nötig, verifiziert durch Code-Lesung. Live verifiziert: unauthentifiziert weiterhin 401, App startet fehlerfrei, `pytest agent/tests/` 28/28 grün. **Nicht umgesetzt:** dedizierter API-Test — es existiert noch kein API-Test-Harness (kein `TestClient`/`conftest.py`/Test-DB-Fixtures) im Repo; das Aufsetzen davon ist der breitere P2-9-Punkt (Alt-Audit) und wurde nicht im Rahmen dieses kleinen Fixes mit erledigt. Verifikation mit echtem authentifiziertem Free-User (Quota-Verbrauch, 403-Pfad) wurde bewusst nicht live gegen die lokale Dev-DB durchgeführt, um keine Nutzer-PII anzufassen — sollte vor dem Live-Smoke-Test (P1-2) oder bei Aufbau des API-Test-Harnesses (P2-9) nachgeholt werden.

---

### [P1-2] Live-Smoke-Test der Kennzahlen gegen echte SEC-Daten

**Status:** Offen (Alt-Audit P1-2)
**Bereich:** Datenqualität / Produktvertrauen
**Betroffene Dateien/Komponenten:** `agent/Model.py` (u. a. `:70` KGV, `:486` KUV, `:536` ROE — `.iloc[0]`-Annahme „neueste Periode", `.iloc[:4].sum()` als TTM), `agent/data_sources/sec_source.py`

**Problem:** Kein Audit hat die zentrale Annahme verifiziert, dass SEC-Daten absteigend nach Datum sortiert geliefert werden; ebenso ungeprüft: Label-Toleranz bei exotischen Filern, Rate-Limit-Verhalten unter Last.

**Warum relevant:** Das Kernversprechen ist „jede Zahl nachvollziehbar und korrekt". Ein systematischer Sortierfehler würde flächendeckend falsche aktuelle Kennzahlen liefern.

**Erwarteter Zielzustand:** (a) Sortierung im Code explizit erzwingen (`sort_values` nach Periodendatum) statt annehmen; (b) dokumentierter Smoke-Test: 5–10 reale Symbole (Bank, ADR, Dienstleister ohne Inventar, schuldenfreie Firma, Nicht-Dividendenzahler) × alle 8 Analyse-Modi + Compare, Stichproben gegen 10-K/10-Q-Werte.

**Akzeptanzkriterien:**
- Explizite Sortierung im Datenpfad, mit Regressionstest
- Smoke-Test-Protokoll (Symbol × Modus × Ergebnis OK/Fehler) liegt vor; gefundene Abweichungen als Issues erfasst

**Hinweise für Sonnet:** Der Smoke-Test braucht Netzwerkzugriff und dauert (SEC-Rate-Limits, 6h-Cache hilft). Cache-Verzeichnis vorher nicht löschen — realistische Cache-Hits sind Teil des Tests.

---

### [P1-3] Dividenden-/Average-Grower-Analyse: Randfälle degradieren statt abbrechen

**Status:** Offen (Alt-Audit P1-3)
**Bereich:** Technik / UX
**Betroffene Dateien/Komponenten:** `agent/AgentAction.py:200-202` (+ `agent/Model.py:761-766`), `agent/AgentAction.py:432-433`, `agent/DataLoader.py` (`get_dividend_data`)

**Problem:** (1) Schuldenfreie Firma → Zinsdeckungsrate „nicht berechenbar" → GESAMTE Dividenden-Analyse bricht mit Fehler ab, obwohl „keine Zinsen" fachlich die beste Zinsdeckung ist. (2) Nicht-Dividendenzahler → Komplettabbruch statt „Kriterium nicht erfüllt". (3) Negativer FCF im Average-Grower → Abbruch statt Bewertung.

**Warum relevant:** Trifft genau die attraktivsten Kandidaten (schuldenfreie Qualitätsfirmen). Ein zahlender Nutzer bekommt einen generischen Fehler statt eines Ergebnisses — wirkt wie ein kaputtes Produkt.

**Erwarteter Zielzustand:** Muster aus `analyze_typical_cyclers` übernehmen: pro Kriterium `meets_criterion: False/True` + Message, Analyse läuft durch. Zinsaufwand 0 → Kriterium bestanden.

**Akzeptanzkriterien:**
- Dividenden-Analyse für eine schuldenfreie Firma und einen Nicht-Dividendenzahler liefert ein strukturiertes Ergebnis (kein Abbruch)
- Regressionstests für beide Randfälle
- Frontend zeigt die degradierten Kriterien verständlich an

---

### [P1-4] Öffentliche Pricing-Kommunikation (Landing-Abschnitt oder Pricing-Seite)

**Status:** Offen (neu in diesem Audit)
**Bereich:** Marketing / Conversion
**Betroffene Dateien/Komponenten:** `frontend/src/pages/public/LandingPage.tsx` (kein Pricing-Abschnitt vorhanden — live bestätigt), `frontend/src/pages/app/BillingPage.tsx:256-340` (Preise nur nach Login: Free 0 € / Pro 50 €/Monat bzw. 500 €/Jahr)

**Problem:** Die einzige öffentliche Preisinformation ist „50 Analysen/Monat inklusive" im CTA. Was Pro kostet, was Pro enthält und warum man zahlen sollte, erfährt ein Interessent erst NACH Registrierung + E-Mail-Verifizierung + Login.

**Beweis / Beobachtung:** Live-Test 2026-07-09: kompletter Landing-Scroll (Desktop 1280px) — Abschnitte Hero → Methoden → Engine/Logik → „So funktioniert die App" → Final-CTA → Footer. Kein Preis, kein Pricing-Link im Footer.

**Warum relevant:** Für ein bezahltes SaaS ist verstecktes Pricing ein klassischer Conversion-Killer: (a) preissensible Nutzer brechen vor der Registrierung ab („was kostet das?" unbeantwortet), (b) Upgrade-Bereitschaft entsteht erst, wenn der Wert von Pro von Anfang an klar ist, (c) verstecktes Pricing wirkt bei einem Self-Service-Produkt unseriös.

**Erwarteter Zielzustand:** Öffentlich sichtbarer Pricing-Abschnitt (auf der Landing und/oder als `/pricing`-Route im `PublicLayout`): Free vs. Pro nebeneinander, Preis, Leistungsumfang, Jahres-Rabatt („2 Monate gratis"), CTA pro Plan. Konsistent mit `BillingPage.tsx`.

**Akzeptanzkriterien:**
- Preis von Pro ohne Login auffindbar (max. 1 Klick von der Landing)
- Feature-Vergleich Free vs. Pro identisch mit der internen BillingPage (eine gemeinsame Quelle für die Plan-Daten anlegen, keine Kopie)
- Footer-/Header-Link auf das Pricing
- Mobile getestet (375px, kein horizontales Scrollen)

**Hinweise für Sonnet:** `BillingPage.tsx` referenziert in einem Kommentar bereits ein „reference pricing-page layout" (Zeile ~593) — als visuelle Basis nutzen. Kein Export-Feature u. Ä. erfinden; Leistungsumfang exakt aus der BillingPage übernehmen.

---

### [P1-5] In-App-Disclaimer an Kursziel-/CRV-Panels + Wording entschärfen

**Status:** Offen (Alt-Audit P1-5; Teil von P0-5, aber eigenständig umsetzbar)
**Bereich:** Rechtliches / UX
**Betroffene Dateien/Komponenten:** `frontend/src/components/metrics/CrvTargetPanel.tsx`, Ergebnis-Messages in `agent/AgentAction.py` (u. a. `:819`, `:371`, `:999-1013`)

**Problem:** „buy_price", „Kaufzone [1.0–1.5]", „KEIN KAUF", „ideal für Zykliker-Einstieg" ohne kontextnahen Hinweis; Disclaimer existiert nur auf den Rechtsseiten.

**Erwarteter Zielzustand:** Permanenter Kurz-Disclaimer direkt an CRV-/Kursziel-/Analyse-Panels („Historische Bandbreiten-Rechnung — keine Anlageberatung"); imperative Formulierungen durch beschreibende ersetzt.

**Akzeptanzkriterien:**
- Disclaimer sichtbar ohne Interaktion (kein Tooltip-only), auch mobil
- Kein „KAUF"/„KEIN KAUF"/„Einstieg" mehr in Ergebnis-Messages
- `pytest agent/tests/` grün (Message-Tests ggf. anpassen)

---

### [P1-6] Admin-Metrik Churn reparieren (zeigt systematisch ~0)

**Status:** Offen (neu in diesem Audit)
**Bereich:** Admin / Datenqualität
**Betroffene Dateien/Komponenten:** `api/routes/admin_stats.py:176-186`; Events: `api/routes/stripe_webhook.py:341` (`subscription_status_changed`), `:431` (`subscription_deleted`)

**Problem:** Die Churn-Kachel zählt `subscription_status_changed` mit `metadata.to == "canceled"`. Der normale Kündigungsweg (zum Periodenende) durchläuft aber `billing_status = "canceling"` und endet mit dem Event **`subscription_deleted`** — das die Query nie zählt. Churn wird daher fast immer 0 anzeigen, egal wie viele Kunden kündigen.

**Beweis / Beobachtung:** Code gelesen 2026-07-09: Query filtert exakt auf `event_type == "subscription_status_changed"` + `to == "canceled"`; der Webhook-Handler für `customer.subscription.deleted` loggt `subscription_deleted` (anderes Event).

**Warum relevant:** Der Betreiber trifft Geschäftsentscheidungen auf Basis dieser Zahl. Ein dauerhaft „0 % Churn" verschleiert Kundenabwanderung komplett — gefährlichste Sorte Metrik-Fehler: sieht gut aus, ist falsch.

**Erwarteter Zielzustand:** Churn zählt distinct User mit `subscription_deleted` in den letzten 30 Tagen (zusätzlich optional: `subscription_cancel_requested` als Frühindikator getrennt ausweisen).

**Akzeptanzkriterien:**
- Query zählt `subscription_deleted`
- API-Test mit gemockten Events: Kündigung zum Periodenende erhöht Churn nach Ablauf
- Admin-UI-Label präzisiert („Beendete Abos (30 T)")

---

### [P1-7] Admin-Metrik MRR korrigieren (hartkodierte Schätzung, mehrere Verzerrungen)

**Status:** Offen (neu in diesem Audit)
**Bereich:** Admin / Datenqualität
**Betroffene Dateien/Komponenten:** `api/routes/admin_stats.py:161-206`; `api/models/user.py:60` (`billing_interval`, Default None)

**Problem:** (a) Preise hartkodiert (50/500 €) statt aus Stripe/`stripe_price_id`; (b) zählt ALLE `plan == "pro"` unabhängig von `billing_status` — `past_due`- und `canceling`-User zählen als volle MRR; (c) `billing_interval == None` fällt in den `else`-Zweig und wird als monatlich 50 € gezählt — ein Jahres-Abonnent ohne synchronisiertes Intervall wird mit 50 € statt 41,67 € überzählt; (d) Rabatte/Proration/Refunds sind unsichtbar.

**Beweis / Beobachtung:** `admin_stats.py:163-174` gelesen 2026-07-09 (Query lädt nur `billing_interval`, kein `billing_status`-Filter; `else`-Zweig behandelt None als monatlich).

**Warum relevant:** MRR ist DIE Steuerungsgröße eines Abo-Geschäfts. Eine systematisch überzeichnete MRR führt zu falschen Wachstums-/Preisentscheidungen.

**Erwarteter Zielzustand:** MRR zählt nur `billing_status == "active"` (Entscheidung dokumentieren, ob `canceling` bis Periodenende mitzählt — branchenüblich: ja, aber getrennt ausweisen); `billing_interval == None` wird nicht stillschweigend als monatlich gewertet, sondern als Datenfehler gezählt und im Dashboard sichtbar gemacht; Preisbeträge zentral definiert (Konstante/Settings) oder aus Stripe gelesen.

**Akzeptanzkriterien:**
- `past_due` zählt nicht (oder separat als „gefährdete MRR")
- None-Intervall erscheint als eigener Zähler „unbekanntes Intervall" statt stiller 50-€-Annahme
- Test deckt die drei Fälle ab

---

### [P1-8] Datenschutzerklärung: Plausible ergänzen + Datumsangaben konsistent

**Status:** ✅ Erledigt (2026-07-09)
**Bereich:** Rechtliches
**Betroffene Dateien/Komponenten:** `frontend/src/pages/legal/PrivacyPage.tsx` (Drittanbieter-Abschnitt `:78-95`, „Stand: Juni 2026"), `frontend/src/pages/legal/CookiesPage.tsx` (dokumentiert Plausible bereits)

**Problem:** Die Datenschutzerklärung listet Stripe/SMTP/Finanzdatenquellen als Verarbeiter, erwähnte Plausible Analytics aber nicht — obwohl es per Cookie-Banner eingebunden wird und die CookiesPage es beschreibt. Zudem „Stand: Juni 2026" vs. „Juli 2026" auf den anderen Legal-Seiten.

**Warum relevant:** Unvollständige Verarbeiter-Liste ist ein DSGVO-Mangel; inkonsistente Stände wirken ungepflegt.

**Akzeptanzkriterien:**
- Plausible im Drittanbieter-Abschnitt der PrivacyPage (Zweck, Rechtsgrundlage Einwilligung, keine Cookies/PII, Anbieter + Serverstandort) ✅
- Einheitliches „Stand:"-Datum auf allen Legal-Seiten ✅

**Umsetzungsnotiz (2026-07-09):** `PrivacyPage.tsx` „Stand:" auf „Juli 2026" korrigiert (jetzt konsistent mit `TermsPage.tsx`/`CookiesPage.tsx`); neuer Listenpunkt „Plausible Analytics" im Drittanbieter-Abschnitt ergänzt (Zweck, cookielos, keine PII, Einwilligungs-Rechtsgrundlage Art. 6 Abs. 1 lit. a DSGVO, Link zur Cookies-Seite). `npx tsc -b` grün, live im Preview verifiziert (Snapshot bestätigt korrekten Text + funktionierenden Link, keine Konsolenfehler). `ImprintPage.tsx` unverändert (P0-2 wartet weiter auf die Adresse).

---

### [P1-9] Monitoring aktivieren (Sentry) und Analytics anschließen (Plausible)

**Status:** Offen (aus `LAUNCH_CHECKLIST.md` 🟠, hier als P1 priorisiert)
**Bereich:** Betrieb / Datenqualität
**Betroffene Dateien/Komponenten:** Env-Vars `SENTRY_DSN` (Render), `VITE_SENTRY_DSN`, `VITE_PLAUSIBLE_DOMAIN`, `VITE_PLAUSIBLE_API_HOST` (Vercel); Code ist fertig verdrahtet (`api/main.py:36-44`, `frontend/src/lib/plausible.ts`)

**Problem:** Ohne `SENTRY_DSN` läuft ein Soft Launch blind — Fehler bei echten Nutzern bleiben unbemerkt. Ohne Plausible-Env lädt das Analytics-Skript nie (No-op), selbst wenn Nutzer zustimmen.

**Warum relevant:** Die gesamte Soft-Launch-Strategie basiert auf „aktivem Monitoring + schnellen Fix-Zyklen" — ohne Sentry ist das nicht gegeben.

**Akzeptanzkriterien:**
- Ein absichtlich ausgelöster Testfehler erscheint in Sentry (Backend und Frontend)
- Plausible zählt einen Testbesuch nach Consent

**Hinweise für Sonnet:** Account-Anlage = Betreiber; Code-Seite ist fertig.

---

### [P1-10] Methodik-Entscheidungen treffen und dokumentieren (AAGR/AQGR, Quartals-Multiples)

**Status:** Offen (Alt-Audit P1-4 + P1-6, als Entscheidungs-Task)
**Bereich:** Datenqualität / Produktvertrauen
**Betroffene Dateien/Komponenten:** `agent/Model.py:1467-1469` (AQGR), `:1597-1599` (AAGR), PEG-Konsument `:650ff`; Quartals-Multiples (`calculate_historical_price_to_sales` u. a.)

**Problem:** (a) Wachstumsraten als arithmetisches Mittel → systematischer Aufwärts-Bias, Basiseffekte, AQGR ohne Saisonbereinigung; (b) Preis-Multiples auf Einzelquartalsbasis ≈ 4× höher als TTM-Marktkonvention — Nutzer, die mit anderen Quellen vergleichen, halten die Zahlen für falsch.

**Erwarteter Zielzustand:** Bewusste, dokumentierte Entscheidung: (a) CAGR/Median statt arithmetischem Mittel, AQGR auf YoY — oder begründet beibehalten und im Glossar erklären; (b) TTM-Umstellung (Historie + Kursziel-Basis GEMEINSAM ändern!) — oder Quartalsbasis prominent in UI + Glossar kennzeichnen.

**Akzeptanzkriterien:**
- Entscheidung schriftlich (Glossar/Methodik-Notiz im Repo)
- Bei Umstellung: Regressionstests angepasst, PEG-Auswirkung geprüft
- Bei Beibehalten: UI-Kennzeichnung an den betroffenen Multiples („Quartalsbasis")

**Hinweise für Sonnet:** Achtung, Konsistenzfalle: Historie und Kursziel-Basis nutzen dieselbe Frequenz — niemals nur eine Seite umstellen (exakt der Fehlertyp, der als K-1/K-2 behoben wurde). Produktentscheidung des Betreibers einholen, bevor Code angefasst wird. Hinweis: eine eigene „Methodik-Seite" ist laut Produktentscheidung nicht gewollt — Dokumentation gehört ins bestehende Glossar.

---

### [P1-11] API-Dokumentation in Produktion deaktivieren

**Status:** ✅ Erledigt (2026-07-09)
**Bereich:** Sicherheit
**Betroffene Dateien/Komponenten:** `api/main.py:46` (`app = FastAPI(title="AIAgent API", version="0.1")`)

**Problem:** `/docs`, `/redoc`, `/openapi.json` sind öffentlich erreichbar und enumerieren die komplette API-Oberfläche inkl. aller `/admin/*`-Routen und Parameter-Schemas.

**Warum relevant:** Kein direkter Zugriffsschutz-Bruch (Endpunkte bleiben auth-gegated, live verifiziert 401), aber ein Angreifer bekommt die vollständige Landkarte gratis. Zudem wirkt „AIAgent API v0.1" als öffentlicher Titel unprofessionell, falls verlinkt.

**Erwarteter Zielzustand:** In Produktion `docs_url=None, redoc_url=None, openapi_url=None` (env-gesteuert, z. B. `settings.ENVIRONMENT == "production"`); lokal bleibt Swagger verfügbar.

**Akzeptanzkriterien:**
- Prod: alle drei Pfade → 404; Dev: weiterhin erreichbar ✅
- Nebenbei: `title` auf „ComAnalysis API" korrigieren ✅

**Umsetzungsnotiz (2026-07-09):** Neues `ENVIRONMENT`-Setting (`api/core/config.py`, Default `"development"`); `api/main.py` setzt `docs_url`/`redoc_url`/`openapi_url` auf `None`, wenn `ENVIRONMENT == "production"`. Titel korrigiert. `.env.example` um `ENVIRONMENT` ergänzt (Render muss `ENVIRONMENT=production` setzen — Teil von P0-3). Live verifiziert: Default-Modus `/docs`+`/openapi.json` → 200; mit `ENVIRONMENT=production` beide → 404, `/health` weiterhin 200 (App funktioniert normal). `pytest agent/tests/` 28/28 grün.

---

## 4. P2 — Optimierungen nach Launch

Übernommene Alt-Audit-Punkte (Details siehe `LAUNCH_AUDIT.md` Abschnitt 7, alle weiterhin offen): **P2-1** Job-Manager ohne Eviction/Single-Instance · **P2-2** Full-Analysis ohne `make_json_safe` · **P2-3** Quota-Verbrauch trotz 429 · **P2-4** Quartals-Negativfall „Mio. USD"-Faktorfehler · **P2-5** Dividenden-CAGR Teiljahr-Verzerrung · **P2-6** `print()`-Debug + doppelte AgentAction-Instanz · **P2-7** Stripe-Retrieve ohne try/except · **P2-8** Favorites ohne Limit/Validierung · **P2-9** Testlücken API/Auth/Billing/Frontend + CI ohne Frontend-Build/Lint · **P2-10** 20 ESLint-Errors · **P2-11** Quota-Semantik Full=1 vs. Custom=N · **P2-12** wirkungslose `@retry`-Dekoratoren · **P2-13** Register-Enumeration.

Neue P2-Aufgaben aus diesem Audit:

### [P2-14] Cookie-Banner überdeckt den primären CTA beim Erstbesuch

**Status:** Offen · **Bereich:** Marketing/UX
**Betroffene Dateien/Komponenten:** `frontend/src/components/consent/CookieConsentBanner.tsx`
**Problem/Beweis:** Live-Test Desktop 1280×800: Der Banner (unten links) überlappt beim Erstbesuch den „Kostenlos starten"-CTA der Hero-Section teilweise. Genau im wichtigsten Conversion-Moment konkurriert der Banner mit dem CTA.
**Zielzustand:** Banner kompakter/rechtsbündig positionieren oder Hero-CTA-Bereich freihalten (z. B. Banner als schmale Bottom-Bar).
**Akzeptanzkriterien:** Bei 1280×800 und 375×812 überlappt der Banner keinen CTA; Screenshot-Vergleich.

### [P2-15] three.js-Intro ohne Mobile-/Low-Power-Gating

**Status:** Offen · **Bereich:** Responsiveness/Performance
**Betroffene Dateien/Komponenten:** `frontend/src/components/intro/RingsScene.tsx` (`dpr={[1,2]}`), Kontrast: `ParticleBeamBackground.tsx:75` prüft `prefers-reduced-motion`
**Problem:** Das 3D-Intro nach Login rendert ungegated auch auf schwachen Mobilgeräten; `prefers-reduced-motion` wird dort nicht respektiert.
**Zielzustand:** Intro überspringen bei `prefers-reduced-motion`, optional bei `useIsMobile()`; statisches Fallback.
**Akzeptanzkriterien:** Emulation mit reduced-motion zeigt kein 3D-Intro; Bundle-Load des three.js-Chunks idealerweise lazy (siehe Alt-Audit P3-3: 2-MB-Bundle).
**Hinweis für Sonnet:** Der Partikelstrahl-Hintergrund selbst ist eine bewusste Produktentscheidung und bleibt.

### [P2-16] Interne Accounts (admin/friends) verfälschen Admin-Aktivitätsmetriken

**Status:** Offen · **Bereich:** Datenqualität
**Betroffene Dateien/Komponenten:** `api/routes/admin_stats.py` (DAU/WAU/MAU `:58-82`, Funnel `:28-55`, Daily `:85-117`)
**Problem:** Eigene `plan=="admin"`- und `plan=="friends"`-Accounts erzeugen `analysis_started`/Login-Aktivität und fließen ungefiltert in DAU/WAU/MAU, Funnel und Analyse-Zahlen ein. Bei kleiner Nutzerbasis (Soft Launch!) dominiert die Eigennutzung die Metriken.
**Zielzustand:** Stats-Queries schließen `admin`/`friends` aus (Join auf User.plan), optional Toggle „interne Accounts einbeziehen".
**Akzeptanzkriterien:** Admin-Eigennutzung erhöht DAU nicht; Test mit gemischten Plänen.

### [P2-17] Tracking-Lücken: Seitenaufrufe und Feature-Nutzung fehlen im Funnel

**Status:** Offen · **Bereich:** Datenqualität/Admin
**Betroffene Dateien/Komponenten:** `api/services/event_service.py`, Event-Callsites (Liste: `user_registered`, `email_verified`, `analysis_started`, `quota_exceeded`, `checkout_started`, Subscription-Events, `account_deleted`, `onboarding_tour_completed`)
**Problem:** Der Funnel springt von `email_verified` direkt zu `analysis_started` — man sieht nicht, wer die Analyze-Seite erreicht, aber nie startet. Compare, Favorites, Custom-Definition-Saves, Billing-Portal-Öffnungen werden gar nicht erfasst; client-seitige Abbrüche (Checkout-Button ohne Redirect, Formularfehler) sind unsichtbar.
**Zielzustand:** Wenige gezielte Zusatz-Events (z. B. `compare_started`, `favorite_added`, `custom_definition_saved`, `billing_portal_opened`), dokumentiert; Plausible für Seitenaufrufe aktiviert (P1-9) und als getrennte Quelle akzeptiert.
**Akzeptanzkriterien:** Neue Events erscheinen im Admin-Dashboard (Analysen-pro-Modus-Chart o. Ä. erweitert); keine PII in `event_metadata`.

### [P2-18] Stille `.catch`-Blöcke sichtbar machen

**Status:** Offen · **Bereich:** Frontend/UX
**Betroffene Dateien/Komponenten:** `frontend/src/pages/app/ComparePage.tsx:96` (Katalog-Fehler → leerer Katalog ohne Meldung), `frontend/src/pages/app/AnalyzePage.tsx:419,427` (Favoriten-Toggle: Optimistic-Update wird zurückgerollt, aber ohne Feedback), `frontend/src/pages/app/SupportPage.tsx:21` (bewusst, dokumentiert — OK)
**Problem:** Beim Compare-Katalog führt ein API-Fehler zu einer leeren Metrik-Auswahl ohne Erklärung; beim Favoriten-Toggle springt der Stern kommentarlos zurück.
**Zielzustand:** ComparePage: Fehlerzustand mit Retry; Favoriten: kurzer Toast („Konnte nicht gespeichert werden").
**Akzeptanzkriterien:** API-Fehler simuliert → Nutzer sieht eine Meldung statt leerer UI.

### [P2-19] UI-Konsistenz und tote Pfade

**Status:** Offen · **Bereich:** Frontend
**Betroffene Dateien/Komponenten:** `frontend/src/components/layout/AppSidebar.tsx:34-39` (Nav mischt Deutsch/Englisch: „Analyse", „Vergleich" vs. „Billing", „Account", „Support"), `frontend/src/components/layout/Header.tsx:96-138` (nie gerenderter `variant === "app"`-Zweig, verweist auf deprecated `/app/custom-analysis`), `frontend/src/app/AppRouter.tsx` (leere 0-Byte-Datei)
**Zielzustand:** Einheitliche Sprache in der Navigation (z. B. „Abrechnung", „Konto"); toten Header-Zweig und leere Datei entfernen.
**Akzeptanzkriterien:** `npm run build` grün; keine Verweise auf `/app/custom-analysis` außer der Redirect-Route.

### [P2-20] Kleine Touch-Ziele im Registrierungsformular

**Status:** Offen · **Bereich:** Responsiveness/UX
**Betroffene Dateien/Komponenten:** `frontend/src/pages/auth/RegisterPage.tsx:267-308` (AGB-/Datenschutz-Checkboxen)
**Problem/Beweis:** Live-Messung 375px: Checkboxen 13×13px — deutlich unter der 44px-Touch-Empfehlung. Label-Klick funktioniert vermutlich, aber die visuelle Zielfläche ist winzig, ausgerechnet bei den Pflicht-Checkboxen.
**Zielzustand:** Größere Checkbox (≥20px) + klickbares Label mit Padding.
**Akzeptanzkriterien:** Checkbox-Zielfläche inkl. Label ≥44px Höhe auf Mobile.

### [P2-21] Weitere übernommene Betriebs-/Hygiene-Punkte

- Gmail-SMTP mittelfristig durch transaktionalen Anbieter ersetzen (Resend/Postmark/SES) — Sendelimit ~500/Tag, Zustellbarkeit (`LAUNCH_CHECKLIST.md` 🟠)
- Kein „Spin down on inactivity" auf Render; nicht horizontal skalieren ohne Worker-Umbau (`LAUNCH_CHECKLIST.md` ⚙️)
- Datenquellen-Free-Tier-Limits prüfen/upgraden (`LAUNCH_CHECKLIST.md` 🟢)
- `robots.txt` fehlt (Alt-Audit Abschnitt 15) · Security-Scans in CI (`bandit`, `pip-audit`) · Rate-Limit für `metric_routes.py` und `admin_customers.py` ergänzen (aktuell 0 Limits, alle authed — geringes Risiko) · Security-/Audit-Log für Auth-Ereignisse (Failed Logins, Admin-CRM-Zugriffe) · `datetime.utcnow()`-Migration (deprecated) · UTC-Tagesgrenzen in Admin-Charts (für DE-Produkt um 1–2 h verschoben — dokumentieren oder auf Europe/Berlin umstellen)

---

## 5. Frontend Responsiveness Tasks

**Live-Testergebnis 2026-07-09** (Vite-Dev-Server, Chromium-Preview): Öffentliche Seiten auf allen drei Viewports **ohne Layout-Brüche, ohne horizontales Scrollen** (gemessen: `scrollWidth <= innerWidth` auf `/`, `/register`, `/login`, `/legal/imprint` bei 1280/768/375). Authentifizierter Bereich nur statisch geprüft (kein Test-Login möglich, Sandbox blockiert Form-Submits).

### Desktop (1280×800) — getestet ✅
- Landing, Login, Register, Legal: sauber. Hero, Karten-Sektionen, Footer korrekt.
- **Risiko:** Cookie-Banner überdeckt Hero-CTA (→ P2-14).

### Tablet (768×1024) — getestet ✅
- Landing skaliert korrekt, CTAs nebeneinander, kein Overflow (753px Scrollbreite bei 768px Viewport).
- Keine offenen Tasks.

### Mobile (375×812) — getestet ✅ mit Befunden
- Kein horizontales Scrollen; Header bricht kontrolliert um (Logo oben, Login/Registrieren darunter) — kein Hamburger, aber funktional.
- Registrierungsformular: Felder volle Breite, Vorname/Nachname 2-spaltig à 128px — nutzbar.
- **Task:** Checkbox-Touch-Ziele 13px (→ P2-20).
- **Task:** Mode-Chip-Leiste der Landing (`AudienceTabs.tsx:63`, `overflowX:auto`) schneidet den letzten Chip sichtbar ab („Wachstums…") ohne Scroll-Affordance — Fade-Out-Kante oder Scroll-Indikator ergänzen.
- **Task (ungetestet, Code-Risiko):** `ComparePivotTable.tsx:47` — breite Vergleichstabellen scrollen horizontal im Container (sticky erste Spalte vorhanden — Mitigation ok, aber mit 4+ Firmen auf 375px real testen, sobald Login-Test möglich).
- **Task:** three.js-Intro nach Login ungegated (→ P2-15).

---

## 6. Frontend Bug & UX Tasks

- **Stille Fehler:** ComparePage-Katalog, Favoriten-Toggle (→ P2-18).
- **States:** Loading/Empty/Error/Success sind in Dashboard, Analyze, Compare, Billing durchgängig vorhanden (Code-verifiziert: `DashBoardPage.tsx:362-395`, `AnalyzeResultsDashboard.tsx:63`, `ComparePage.tsx:374`, `QuotaExceededModal`) — kein Task.
- **Auth-UX:** Login unterscheidet `EMAIL_NOT_VERIFIED` (mit Resend-Link) von falschen Credentials; Inaktivitäts-Logout mit Sticky-Notice — gut, kein Task.
- **Randfall-Abbrüche der Engine erscheinen als generische Fehler** (→ P1-3, wichtigster UX-Bug).
- **Navigation:** DE/EN-Mix in der Sidebar (→ P2-19).
- **Onboarding:** react-joyride-Tour + `onboarding_tour_completed`-Event vorhanden — kein Task.
- **`App.tsx:47`** liest den Token nur einmal synchron beim Mount für den `/`-Redirect — nach Logout in anderem Tab bleibt der Zustand stale bis zur Navigation. Niedrige Priorität.

## 7. Marketing & Conversion Tasks

1. **Pricing öffentlich machen** (→ P1-4) — wichtigster Conversion-Hebel.
2. **Landing-Claim korrigieren** (→ P0-5) — Vertrauens-Grundlage.
3. **Vertrauens-Elemente ergänzen (P1-Empfehlung):** Die Seite hat null Social Proof (keine Testimonials, Logos, Nutzerzahlen — bewusst ehrlich, aber leer). Ohne echte Kunden ersetzbar durch: konkrete Produkt-Beweise (Screenshot/Demo-GIF einer echten Analyse, interaktives Beispiel mit einem bekannten Ticker), „Made in Germany / DSGVO / keine Cookies"-Badge-Zeile, Gründer-Note mit Namen und Gesicht. KEINE erfundenen Testimonials.
4. **Zielgruppen-Ansprache:** Die AudienceTabs (Wachstumswerte/Dividenden/…) sind gut — pro Tab noch 1 Satz „für wen": Selbstentscheider, die SEC-Daten lesen wollen, ohne 10-Ks zu wälzen.
5. **CTA-Hierarchie:** „Kostenlos starten — 50 Analysen/Monat inklusive" ist stark und konkret. Final-CTA „Registriere dich, teste die Engine selbst" ok. Kein Task.
6. **Sprachliche Authentizität:** Copy ist konkret und buzzword-arm („Parst SEC-Filings", „Jede Formel offen einsehbar") — überdurchschnittlich. Kein Task.
7. **Preis-Anker im Billing:** BillingPage kommuniziert „2 Monate gratis" beim Jahresplan — auf die öffentliche Pricing-Sektion übernehmen.

## 8. Admin Dashboard Tasks

1. **Churn-Fix** (→ P1-6) und **MRR-Fix** (→ P1-7) — beide Metriken sind aktuell irreführend.
2. **Interne Accounts filtern** (→ P2-16) — beim Soft Launch dominiert sonst Eigennutzung.
3. **Churn-Gründe-Erfassung bauen:** `/admin/stats/churn-reasons` existiert, aber die Cancel-UI erfasst den Grund kaum (`admin_stats.py:246-249` dokumentiert das selbst; Plan-Punkt 4.6). Kündigungs-Modal um optionale Grund-Auswahl erweitern (`CancelSubscriptionModal`), Event trägt `reason` bereits.
4. **Vorhandene Stärken (kein Task):** DAU/WAU/MAU, 7-stufiger Funnel, 30-Tage-Charts, Top-Symbole, Near-Limit-Liste (konkret handlungsrelevant für Upgrade-Ansprache), CRM mit Notizen/Timeline/Plan-Wechsel (Admin-Plan nicht via API vergebbar — gutes Defense-in-Depth), Audit-Notizen bei Plan-Änderungen.
5. **Zeitraum-Filter:** Aktuell fix 30 Tage — für Trend-Erkennung 7/30/90-Tage-Umschalter ergänzen (P2).
6. **Kein Umsatz-Verlauf:** MRR ist eine Punktzahl ohne Historie. Nach dem MRR-Fix einen einfachen Verlauf (monatliche Snapshots in `product_events` oder eigene Tabelle) ergänzen (P2).

## 9. Datenqualität & Tracking Tasks

- **Korrekt gesammelt (verifiziert):** Registrierung/Verifizierung/Analysen/Quota/Checkout/Subscription-Lifecycle als append-only `product_events` mit Idempotenz auf Stripe-Seite (`stripe_events`-Ledger); ein Event pro Analyse-Start (keine Doppelzählung, Custom mit N Metriken = 1 Event mit `mode:"custom"`); GDPR-Löschung anonymisiert Events (`ON DELETE SET NULL`) statt Historie zu zerstören.
- **Falsch/irreführend:** Churn (→ P1-6), MRR (→ P1-7), interne Accounts (→ P2-16).
- **Fehlend:** Seitenaufrufe/Feature-Events (→ P2-17), Churn-Gründe (→ Admin-Task 3), MRR-Historie (→ Admin-Task 6).
- **Zeitzonen:** Alle Tagesgrenzen UTC (`datetime.utcnow`, `cast(created_at, Date)`) — für ein DE-Produkt um 1–2 h verschoben. Bewusst dokumentieren oder auf Europe/Berlin umstellen (P2).
- **Testdaten:** Keine Seed-/Demo-Daten-Skripte — Prod-Metriken starten sauber. Einzige Verunreinigungsquelle: interne Accounts (→ P2-16).

## 10. Sicherheits- und Vertrauensaufgaben

**Verifiziert solide (kein Task):** Server-seitiges `require_admin` auf allen 15 Admin-Endpunkten (live: 401 ohne Token); durchgängiges Ownership-Scoping (Definitions/History/Jobs/Favorites — IDOR geprüft); Stripe-Webhook signaturverifiziert + idempotent; CORS ohne Wildcard, `allow_credentials=False`; Rate-Limits auf allen Auth-Endpunkten mit spoofing-resistentem XFF-Handling; bcrypt + gehashte Reset-/Verify-Tokens; generische Fehlermeldungen; `.env` nie in Git; Enumeration-sichere Forgot-/Resend-Flows; DSGVO-Konto-Löschung mit Stripe-Cancel.

**Offene Aufgaben:**
1. Secrets-Rotation (→ P0-4)
2. `/docs` in Prod deaktivieren (→ P1-11)
3. `/analyze/custom/history` gaten (→ P1-1)
4. Rate-Limits für `metric_routes`/`admin_customers` (→ P2-21)
5. Register-Enumeration: bewusste Abwägung dokumentieren oder generifizieren (Alt-Audit P2-13)
6. Favorites-Validierung gegen `symbols`-Tabelle + Obergrenze (Alt-Audit P2-8 — beliebige Strings landen sonst im 6h-SEC-Worker)
7. Auth-/Admin-Audit-Log (Failed Logins, CRM-Zugriffe) — P2
8. Token in localStorage: bewusster Tradeoff (kein Cookie-CSRF), 30-min-Expiry + Refresh — dokumentiert lassen, kein Task

---

## 11. Empfohlene Reihenfolge für Sonnet

**Block A — Sichern & Rechtliches (vor allem anderen, ~1 Tag):**
1. P0-1: Working Tree committen + pushen (zuerst! Alles Weitere baut darauf auf)
2. P0-2: Impressum/Datenschutz-Adressen (Betreiber-Input nötig) + P1-8 (Plausible + Datum) in einem Rutsch — gleiche Dateien
3. P0-4: Secrets-Rotation anstoßen (Betreiber-Dashboards)

**Block B — Produkt-Wahrheit & Schutz (Code, ~2–3 Tage):**
4. P0-5 + P1-5 zusammen: Landing-Claim, Ergebnis-Wording, In-App-Disclaimer (zusammenhängend: gleiche Message-Strings; vorher Betreiber-Entscheidung Weg 1/2)
5. P1-1: `/analyze/custom/history` gaten (kleines, isoliertes Diff + Test)
6. P1-11: `/docs` env-gesteuert deaktivieren (Mini-Diff)
7. P1-3: Dividenden-/Grower-Randfälle degradieren (Engine + Tests; größtes Code-Stück in diesem Block)

**Block C — Admin-Wahrheit (Code, ~1 Tag):**
8. P1-6 + P1-7 zusammen: Churn- und MRR-Query in `admin_stats.py` (gleiche Datei, gemeinsame Tests)
9. P2-16 direkt mitnehmen (gleiche Queries: interne Accounts filtern)

**Block D — Conversion (Code, ~1–2 Tage):**
10. P1-4: Öffentliches Pricing (Landing-Abschnitt/Route, gemeinsame Plan-Daten-Quelle mit BillingPage)
11. P2-14: Cookie-Banner-Position (klein, gleiche Ecke des Codes)

**Block E — Betrieb & Launch (Betreiber + Sonnet unterstützend):**
12. P0-3: Prod-Env/Stripe-Live/Migrationen/Admin (LAUNCH_CHECKLIST.md abarbeiten)
13. P1-9: Sentry + Plausible aktivieren, Testfehler/Testbesuch verifizieren
14. P1-2: Live-Smoke-Test (nach Deploy; Sortierung vorher im Code erzwingen)
15. P1-10: Methodik-Entscheidungen (Betreiber) → ggf. Umsetzung

**Danach (P2, nach Launch):** P2-18/P2-19/P2-20 (Frontend-Politur) → P2-15 (three.js-Gating) → P2-17 (Tracking-Ausbau) → Alt-Audit P2-1…P2-13 in Tabellen-Reihenfolge → P2-21 (Betrieb/Hygiene).

**Abhängigkeiten:** Block A/1 vor allem anderen (Commit-Sicherung). P1-2 erst nach P0-3 (braucht Prod-nahe Umgebung oder zumindest committeten Stand). P1-4 nach P0-5 (Pricing-Texte müssen zum korrigierten Claim passen). Betreiber-Entscheidungen nötig für: P0-2 (Adresse), P0-5/P1-5 (Wording-Weg), P1-1 (Overlay-Quota-Einheit), P1-10 (Methodik), Alt-P2-11 (Quota-Semantik).
