# ComAnalysis — Launch-Checkliste

Rein organisatorische To-dos vor dem Live-Gang: Accounts anlegen/verbinden, Secrets erneuern, Rechtstexte vervollständigen. Kein Code-Task — das hier sind Dinge, die *du* in Dashboards (Stripe, Vercel, Render, Sentry, Plausible) oder auf Papier (Impressum) erledigen musst.

Stand geprüft: 2026-07-03, gegen den aktuellen Code-Stand (`api/core/config.py`, lokale `.env`, `frontend/.env.example`, Rechtstexte).

---

## 🔴 Blocker — App funktioniert ohne das nicht korrekt

Diese Punkte sind aktuell nachweislich noch auf lokale/Test-Werte gesetzt und würden in Produktion sichtbar kaputtgehen.

- [ ] **`FRONTEND_URL` zeigt noch auf `http://127.0.0.1:5173`** (Backend-Env, Render). Wird in E-Mail-Links verwendet (E-Mail-Verifizierung, Passwort-Reset, Filing-Alerts, Watchlist-Digest) — mit dem aktuellen Wert bekommen Nutzer in Produktion kaputte `localhost`-Links per Mail. Auf die echte Vercel-Domain setzen.
- [ ] **`STRIPE_SUCCESS_URL` / `STRIPE_CANCEL_URL` zeigen noch auf `localhost:5173`**. Nach einer echten Zahlung würde Stripe auf `localhost` weiterleiten. Auf `https://<domain>/app/billing/success` bzw. `.../cancel` setzen.
- [ ] **`CORS_ORIGINS`** (Render-Env) enthält aktuell nur `localhost`-Origins und eine alte Render-URL, aber **nicht** die künftige Vercel-Domain. Ohne das schlägt jeder Login/API-Call vom Frontend mit CORS-Fehler fehl.
- [ ] **Stripe läuft im Test-Modus** (`STRIPE_SECRET_KEY` beginnt mit `sk_test_`). Vor Launch: Stripe-Account auf Live-Modus umstellen (bzw. die Live-Keys aus dem Stripe-Dashboard holen), `STRIPE_SECRET_KEY` und `STRIPE_WEBHOOK_SECRET` als Live-Werte in Render setzen.
- [ ] **Stripe-Preise sind Test-Preise** (`STRIPE_PRICE_ID_PRO_MONTHLY` / `..._YEARLY`). Im Live-Modus müssen die Produkte/Preise (50 €/Monat, 500 €/Jahr) im Stripe-Dashboard **erneut** angelegt werden — Test- und Live-Modus haben getrennte Produktkataloge. Neue `price_...`-IDs in Render eintragen.
- [ ] **Stripe-Webhook-Endpoint** im Stripe-Dashboard auf die Produktions-Backend-URL zeigen lassen (`https://<render-domain>/stripe/webhook` o. ä.) und den dabei generierten Live-Webhook-Secret in `STRIPE_WEBHOOK_SECRET` übernehmen (unterscheidet sich vom Test-Secret).
- [ ] **Datenbank-Migrationen auf der Produktions-DB ausführen** (`alembic upgrade head`) — insbesondere die beiden neuesten aus der aktuellen Session (`108252fde6af` Profilfelder/Consent, `0de6753e5c58` `customer_notes`-Tabelle). Ohne das crasht das Backend beim ersten Request auf die neuen Spalten/Tabellen.
- [ ] **Admin-Account in der Produktions-DB setzen**: `python scripts/set_admin.py gecen.efe1308@gmail.com` **gegen die Produktions-`DATABASE_URL`** ausführen (lokal schon erledigt, aber das ist eine separate Datenbank). Sonst ist `/app/admin` für dich selbst nicht erreichbar.

---

## 🟠 Accounts anlegen bzw. verbinden

- [ ] **Vercel-Projekt** für `frontend/` anlegen, Custom Domain verbinden, folgende Env-Vars im Vercel-Projekt setzen:
  - `VITE_API_BASE_URL` → Produktions-Backend-URL (Render)
  - `SITE_URL` → echte Domain (für Glossar-SEO/Sitemap)
  - `VITE_PLAUSIBLE_DOMAIN` + `VITE_PLAUSIBLE_API_HOST` (siehe Plausible unten)
  - `VITE_SENTRY_DSN` (siehe Sentry unten)
- [ ] **Render-Service** (Backend) mit allen Env-Vars aus `api/core/config.py` befüllen (Liste unten unter "Vollständige Env-Var-Referenz"). **Persistent Disk** einrichten und `CACHE_DIR` auf den gemounteten Pfad zeigen lassen — ohne Persistent Disk leert jeder Deploy den Analyse-Cache (`agent/DataLoader.py`) komplett neu.
- [ ] **Stripe Tax** (optional, aber falls gewünscht): im Stripe-Dashboard aktivieren, dann `STRIPE_AUTOMATIC_TAX=true` setzen (aktuell nicht gesetzt → Default `false`).
- [ ] **E-Mail-Versand**: läuft aktuell über **Gmail-SMTP mit App-Passwort** (`smtp.gmail.com`). Für Produktion mindestens ein **neues, separates App-Passwort** generieren (nicht das lokale Dev-Passwort weiterverwenden). Gmail hat harte Sendelimits (~500 Mails/Tag) und schwankende Zustellbarkeit für Transaktionsmails — als bekannter, bewusst zurückgestellter Punkt: Umstieg auf einen transaktionalen Anbieter (Resend, Postmark, SES) ist empfohlen, aber kein Launch-Blocker bei kleinem Volumen. Wenn du dabei bleibst: Absenderadresse (`EMAIL_FROM`) sollte zur eigenen Domain passen (bessere Zustellbarkeit als eine Gmail-Adresse als Absender).
- [ ] **Sentry**: aktuell **komplett deaktiviert** (`SENTRY_DSN` ist in der lokalen `.env` gar nicht gesetzt → Monitoring läuft still gar nicht). Sentry-Projekt anlegen (ein Projekt fürs Backend, eins fürs Frontend oder ein gemeinsames), DSNs holen und setzen:
  - Backend: `SENTRY_DSN` (Render-Env)
  - Frontend: `VITE_SENTRY_DSN` (Vercel-Env)
- [ ] **Plausible Analytics**: Account auf plausible.io anlegen (oder selbst hosten), Domain dort registrieren, dann in Vercel setzen:
  - `VITE_PLAUSIBLE_DOMAIN` → z. B. `comanalysis.app`
  - `VITE_PLAUSIBLE_API_HOST` → `https://plausible.io` (oder eigene Self-Hosted-URL)
  
  Ohne das lädt das Script nie (No-op), auch wenn Nutzer im Banner zustimmen.
- [ ] **Domain/DNS**: falls noch nicht geschehen, Domain registrieren und auf Vercel zeigen lassen (Vercel-Anleitung: A/CNAME-Records setzen).

---

## 🟡 Rechtliches — echter TODO-Kommentar im Code

- [ ] **Ladungsfähige Anschrift im Impressum eintragen** — [`frontend/src/pages/legal/ImprintPage.tsx`](frontend/src/pages/legal/ImprintPage.tsx) enthält aktuell wortwörtlich die Platzhalter `[Straße und Hausnummer]` und `[PLZ und Ort]` sowie einen Code-Kommentar `TODO vor Launch: ladungsfähige Anschrift eintragen — Pflichtangabe nach § 5 DDG`. **Ohne echte Adresse ist das Impressum nicht rechtskonform.**
- [ ] **Gleiche Adresse in der Datenschutzerklärung ergänzen** — [`frontend/src/pages/legal/PrivacyPage.tsx`](frontend/src/pages/legal/PrivacyPage.tsx) hat unter "Verantwortlicher" noch `[Bitte vollständige Postanschrift ergänzen]`.
- [ ] **USt-ID / Kleinunternehmerregelung klären**: Impressum weist aktuell explizit "keine USt-ID" aus. Falls sich das ändert (z. B. bei Überschreiten der Kleinunternehmergrenze), Impressum aktualisieren — hängt auch mit der Stripe-Tax-Entscheidung oben zusammen.
- [ ] Optional prüfen: Telefonnummer im Impressum ist nach aktueller Rechtsprechung nicht zwingend nötig, wenn E-Mail-Kontakt vorhanden ist (ist er) — kein Handlungsbedarf, nur zur Kenntnis.

---

## 🟢 Datenquellen — API-Keys/Limits prüfen

- [ ] **`ALPHA_VANTAGE_API_KEY`**, **`FRED_API_KEY`**, **`SIMFIN_API_KEY`**: prüfen, ob es sich um Free-Tier-Keys handelt, die unter echtem Nutzer-Traffic an Rate-Limits stoßen könnten. Ggf. auf bezahlte Tarife upgraden, bevor nennenswerter Traffic erwartet wird.
- [x] SEC-EDGAR-Zugriff ist bereits korrekt konfiguriert — der User-Agent (`agent/DataLoader.py:41`) enthält standardmäßig eine echte, identifizierbare E-Mail-Adresse, wie von der SEC gefordert. Kein Handlungsbedarf.

---

## ⚙️ Betriebliche Besonderheit dieses Systems (bitte beachten)

- **Die Hintergrund-Jobs laufen im selben Webprozess**, nicht als separate Render-Worker: `downgrade_worker`, `filing_alert_worker` und `watchlist_digest_worker` (`api/main.py`, gestartet via `asyncio.create_task` beim App-Start) laufen als Endlos-Schleifen direkt im FastAPI-Prozess. Zwei Konsequenzen für die Render-Konfiguration:
  1. **Kein "Spin down on inactivity"** (Render Free Tier) verwenden — sonst laufen Filing-Alerts und der wöchentliche Digest nicht zuverlässig, weil der Prozess bei Inaktivität einschläft.
  2. **Nicht horizontal auf mehrere Instanzen skalieren**, ohne die Worker-Logik anzupassen — sonst liefe z. B. der Filing-Check mehrfach parallel und würde ggf. doppelte Alert-Mails verschicken. Für den Start (eine Instanz) unkritisch, aber vor einem Scale-Up-Schritt merken.

---

## 🧹 Bekannt, nicht blockierend (kann parallel/danach erledigt werden)

- Repo-Hygiene: `API Test.py` (Root) und `agent/cache/*.json` (~320 Dateien) liegen noch im Git-Repo und sollten aus dem Index entfernt werden (`.gitignore` ergänzen + `git rm --cached`).
- CI hat noch keine Security-Scans (`bandit`, `pip-audit`) — bestehender Backlog-Punkt, kein Launch-Blocker.

---

## Vollständige Env-Var-Referenz (Backend, `api/core/config.py`)

Zum Abgleich beim Befüllen der Render-Env-Vars — alle Felder ohne `= Default` sind **Pflicht**, sonst startet das Backend gar nicht:

```
DATABASE_URL                   # Produktions-Postgres (mit Backups!)
SECRET_KEY                     # NICHT den lokalen Dev-Wert wiederverwenden — neuer, zufälliger Produktions-Wert
ACCESS_TOKEN_EXPIRE_MINUTES    # Default 30, i.d.R. unverändert lassen
ALGORITHM                      # Default HS256, unverändert lassen
LOG_LEVEL                      # Default INFO
SQL_ECHO                       # Default false — in Produktion false lassen (Performance/Log-Rauschen)
SENTRY_DSN                     # siehe oben — aktuell leer/deaktiviert
FRONTEND_URL                   # siehe Blocker oben
CORS_ORIGINS                   # siehe Blocker oben
STRIPE_SECRET_KEY              # Live-Modus, siehe Blocker oben
STRIPE_WEBHOOK_SECRET          # Live-Modus, siehe Blocker oben
STRIPE_AUTOMATIC_TAX           # siehe Stripe Tax oben
STRIPE_PRICE_ID_PRO_MONTHLY    # Live-Preis-ID, siehe Blocker oben
STRIPE_PRICE_ID_PRO_YEARLY     # Live-Preis-ID, siehe Blocker oben
STRIPE_SUCCESS_URL             # siehe Blocker oben
STRIPE_CANCEL_URL              # siehe Blocker oben
EMAIL_FROM                     # siehe E-Mail-Versand oben
SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD   # siehe E-Mail-Versand oben
```

Nicht über `Settings` verwaltet, aber ebenfalls per Env-Var nötig:

```
CACHE_DIR                      # Pfad zum Render Persistent Disk (agent/DataLoader.py)
ALPHA_VANTAGE_API_KEY
FRED_API_KEY
SIMFIN_API_KEY
```

## Vollständige Env-Var-Referenz (Frontend, Vercel)

```
VITE_API_BASE_URL              # Produktions-Backend-URL
SITE_URL                       # echte Domain (Build-Zeit, für Glossar/Sitemap)
VITE_SENTRY_DSN                # siehe Sentry oben
VITE_PLAUSIBLE_DOMAIN          # siehe Plausible oben
VITE_PLAUSIBLE_API_HOST        # siehe Plausible oben
```
