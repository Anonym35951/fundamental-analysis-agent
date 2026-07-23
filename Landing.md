# ComAnalysis – Landingpage-Schärfung (Umsetzungsplan für Sonnet 5)

> **So arbeitest du diesen Plan ab:** Tasks **T1–T8** der Reihe nach umsetzen. Priority-1-Tasks
> zuerst (siehe Priorisierung). Nach der Umsetzung die **Verifikation** durchlaufen. Halte dich
> strikt an die **Ehrlichkeits-Leitplanken** – bewirb nur, was im Code existiert.

---

## Context

Ein fachkundiger Professor kritisierte, dass ComAnalysis wie „eine weitere Plattform, die
Aktien für dich analysiert" wirkt und der Nutzen unklar bleibt. Die strategische Antwort ist
**keine** Produkt-Neuentwicklung, sondern eine **schärfere Positionierung der bestehenden
Landingpage**: weg von „Wir analysieren Aktien für dich" → hin zu „Wir geben dir Werkzeuge,
Struktur und Daten, um Unternehmen **selbst** zu analysieren."

Zielgruppe: ambitionierte, selbstständige Privatanleger (zwischen Anfänger und Profi).
Validierungsphase → **kleine Änderungen mit hoher Wirkung**, kein Redesign.

**Verbindliche Entscheidungen des Betreibers:**
1. **Hero-Fokus:** „Selbst analysieren + Struktur" (Säule 1 + 2). Transparenz/Daten werden
   zum **Beleg**, nicht zur Hauptbotschaft.
2. **Datenquelle:** **Vorerst KEINE** Yahoo→Alpha-Vantage-Textänderungen (Yahoo ist im Backend real
   weiterhin Primärquelle). Nur dokumentieren.
3. **Umfang P1:** Copy-Überarbeitung **+ gezielte neue Sections**, unter Wiederverwendung
   bestehender Landing-Komponenten.

Sprache: gesamte Site ist **Deutsch** → alle neuen Texte auf Deutsch.

---

## Ehrlichkeits-Leitplanken (im Code verifiziert – NICHT überschreiten)

| Erlaubt? | Realität im Code |
|---|---|
| ✅ Selbst analysieren, strukturierter Prozess | `AnalyzePage` + 7 Methoden + Custom-Builder + `ComparePage` |
| ✅ Eigene Analyse erstellen & speichern | ABER: = gespeicherte **Kennzahlen-Definitionen** (Name + Metrik-Kombination, CRUD via `saveDefinition`) + Analyse-Historie + Favoriten. **KEIN** freies Thesen-/Notiz-Tool. |
| ✅ Quelle & Stand pro Zahl | `SourceBadge` (Quelle + `as_of` pro Symbol); `DataSourceStatusWidget` (Live-Status) |
| ✅ Formeln offen einsehbar | `metricsConfig.ts` + statisches Kennzahlen-Glossar (`/glossar`) |
| ✅ Kein Black-Box-Score | „Score" = **transparente Kriterien-Zählung** (positive/negative/neutral, `analysisResultUtils.ts::calculateScore`), kein 0–100-Urteil |
| ✅ NYSE + NASDAQ | dynamischer Import aller Stammaktien (`scripts/import_symbols.py`) |
| ✅ Historie | Fundamentaldaten annual/quarterly, Dividenden bis 20 J., densifizierte Market-Cap-Historie |
| ⚠️ „8 vordefinierte Methoden" | Real **7** (`AgentOrchestrator.py`): Wachstumswerte, Dividendenwerte, Average Grower, Typische Zykliker, Zyklische Turnarounds, Optionality, Asset Play → **auf 7 korrigieren** |
| ❌ KI / „AI assists" | **KEINE** KI/LLM im Code. „Agent"-Naming = **regelbasierter** Motor → **NICHT mit KI werben** |

---

## Betroffene Dateien

- `frontend/src/pages/public/LandingPage.tsx` — enthält **alle** Landing-Copy (T1–T7).
- `frontend/src/config/pricingPlans.ts` + `frontend/src/pages/public/PricingPage.tsx` — Pricing-Messaging (T8).
- Wiederzuverwendende Komponenten (kein Neubau): `components/landing/SectionHeading`,
  `FeatureCard`, `StepCard`, `StatReadout`. Styling bleibt Inline-`CSSProperties` + `theme.ts`-Tokens.

---

## Neue Seitenstruktur

1. **Hero** — Konzept A (T1/T2). *HIGH*
2. **Warum ComAnalysis** — 3 Nutzen-Cards (T3). *HIGH*
3. **Eigene Analyse** — NEU, Säule 3 (T4). *HIGH*
4. **So funktioniert die App** — Workflow-Copy (T5). *MEDIUM*
5. **Zahlen verstehen** — NEU, Säule 4, ehrlich (T6). *MEDIUM*
6. **Final CTA** — Wording (T7). *LOW*

FAQ / Screenshots / ChatGPT-Abgrenzung → **P2** (noch nicht bauen).

---

## Tasks

### T1 — Hero-Copy (HIGH) · `LandingPage.tsx`
Gewähltes **Konzept A – „Selbst analysieren + Struktur"**:
- **Eyebrow:** „Fundamentalanalyse · neu gedacht" → **„Für Anleger, die selbst analysieren"**
- **Headline** (3 Zeilen, Zeile 3 behält den Gradient-Span):
  „ALLE FUNDAMENTALDATEN. / EINE QUELLE. / JEDE ZAHL NACHVOLLZIEHBAR."
  → **„ANALYSIERE UNTERNEHMEN / SELBST. / STRUKTURIERT STATT IM DATENCHAOS."**
  (Zeilenumbrüche/Betonung frei anpassen; „STRUKTURIERT…"-Zeile = Gradient-Span.)
- **Subtext** → „ComAnalysis führt dich mit vordefinierten Methoden und einem eigenen
  Logik-Builder durch einen strukturierten Analyseprozess — von den Fundamentaldaten bis zur
  eigenen Einschätzung. Jede Kennzahl mit Quelle und Rechenweg. Die Entscheidung bleibt bei dir."
- **CTAs:** primär „Kostenlos starten — 50 Analyse-Einheiten/Monat inklusive" **KEEP**;
  sekundär „Bereits Zugang? Login" **KEEP**.
- **Checklist-Item 1** („…Kursdaten von etablierten Marktdatenanbietern") **unverändert lassen**
  (bewusst vage → passt zur Datenquellen-Entscheidung). Item 2/3 **KEEP**.

### T2 — Faktenkorrektur „8 → 7 Methoden" (HIGH) · `LandingPage.tsx`
- `StatReadout` `value="8 vordefiniert"` → **`"7 Methoden"`**.
- `FeatureCard` „Vordefinierte Methoden": „Acht etablierte Analysemodi" → **„Sieben etablierte
  Analysemodi"**.
- Prüfe die ganze Datei auf weitere „8"/„Acht"-Vorkommen im Methoden-Kontext.

### T3 — „Warum ComAnalysis": Feature → Nutzen (HIGH) · `LandingPage.tsx`
- Card 1 → Titel **„Bewährte Analysemethoden, sofort einsatzbereit"** / Text „Sieben etablierte
  Denkmodelle — von Wachstum bis Turnaround — statt jede Auswertung manuell aufzubauen."
- Card 2 → Titel **„Deine eigene Analyse-Logik"** / Text „Kombiniere die Kennzahlen, die für dich
  zählen, zu einer wiederverwendbaren Analyse — und speichere sie."
- Card 3 → Titel **„Unternehmen direkt vergleichen"** / Text „Mehrere Unternehmen nebeneinander
  und über die Zeit einordnen — ohne Tool-Wechsel."
- `SectionHeading`-Titel „Eine Plattform statt drei Werkzeuge" **KEEP**; Subtitle auf „selbst zu
  belastbaren Entscheidungsgrundlagen kommen" schärfen.

### T4 — NEUE Section „Eigene Analyse" (HIGH, Säule 3) · `LandingPage.tsx`
Nach „Warum ComAnalysis" einfügen; `SectionHeading` + `FeatureCard`/`StepCard` wiederverwenden.
- Eyebrow „Deine Analyse" · Titel **„Deine Analyse. Deine Einschätzung. Gespeichert an einem Ort."**
- Subtitle: „Konsumiere kein fertiges Urteil — baue deine eigene Sicht auf ein Unternehmen auf.
  Stelle deine eigene Kennzahlen-Logik zusammen, speichere sie und wende sie auf jedes
  Unternehmen wieder an. Deine Analyse-Historie und Favoriten wachsen mit."
- 3 Punkte (nur Vorhandenes): „Eigene Kennzahlen-Definitionen speichern & wiederverwenden" ·
  „Analyse-Historie automatisch" · „Favoriten für schnellen Zugriff".
- **Guard:** NICHT „schreibe deine Investmentthese/Notizen" — existiert nicht.

### T5 — „So funktioniert die App": Subtitle/Workflow (MEDIUM) · `LandingPage.tsx`
- Intern-technischen Subtitle („Nutzer werden von der LandingPage über Login…") ersetzen durch:
  „In wenigen Schritten von der Unternehmenssuche zur eigenen, strukturierten Einschätzung."
- Schritte optional an Ziel-Workflow anlehnen: Unternehmen finden → verstehen → Methode wählen
  oder eigene Logik bauen → mit Quelle & Rechenweg prüfen → vergleichen & speichern.

### T6 — NEUE Section „Zahlen verstehen" (MEDIUM, Säule 4 – EHRLICH) · `LandingPage.tsx`
- Eyebrow „Nachvollziehbarkeit" · Titel **„Verstehe die Zahlen, statt einem Ergebnis zu vertrauen."**
- Subtitle: „Zu jeder Kennzahl siehst du die Quelle und den Stand der Daten sowie die zugrunde
  liegende Formel. Jede Analyse zeigt die einzelnen Kriterien offen — kein einzelner Black-Box-Score."
- Optional: Link auf `/glossar` („Alle Kennzahlen im Glossar").
- **Guard:** KEINE Aussage wie „jeder Datenpunkt bis zum SEC-Filing rückverfolgbar". Erlaubt:
  Quelle+Stand pro Symbol, offene Formeln, Kriterien-Transparenz.

### T7 — Final CTA (LOW) · `LandingPage.tsx`
- KEEP; Titel optional → „Bereit, dein erstes Unternehmen selbst zu analysieren?"

### T8 — Pricing-Messaging (MEDIUM) · `config/pricingPlans.ts` + `PricingPage.tsx`
- **Keine Preisänderung.** Nur Messaging.
- Free-`description`/Pro-`description` weniger „mehr Nutzung", mehr „regelmäßiger, eigenständiger
  Analyseprozess + wachsende eigene Research-Bibliothek". Pro-`description` z. B. „baue dir eine
  unbegrenzte Bibliothek eigener Analyse-Logiken".
- Feature-Zeilen „1 gespeicherte eigene Analyse-Definition" / „Beliebig viele…" **KEEP** (ehrlich).
- `PricingPage`-Subtitle: Nutzen statt nur Limit betonen.

---

## Nicht anfassen

- **Yahoo → Alpha Vantage: JETZT NICHT ändern.** Yahoo ist im Backend real weiterhin Primärquelle
  (Live-Kurse/Aktienzahl/Firmeninfos; Alpha Vantage = Fallback bzw. primär nur Tages-Kursreihen).
  Sichtbare Stellen bleiben, bis das Backend vollständig migriert ist. Betroffen (nur dokumentiert):
  `TermsPage.tsx:38`, `ImprintPage.tsx:54`, `PrivacyPage.tsx:96-97`, `config/metricsConfig.ts:490`
  („yfinance"-Formel), `api/services/data_source_status_service.py:43` (Backend-String →
  `DataSourceStatusWidget`), `SourceBadge` (Payload-Quelle). **Internen Legacy-Code niemals blind
  umbenennen** (`_fetch_yahoo_*`, `yahoo_label`, `_normalize_for_yfinance`, Tests — load-bearing).
- **Keine KI-Werbung.**
- **Kein Redesign**, keine neuen Libraries. Inline-`CSSProperties` + `theme.ts` beibehalten.
- Site-weiter Serif-Fallback bei Headings: bekannt, **out of scope**.

---

## Priorisierung

**PRIORITY 1 (vor dem Marketing-Push):** T1 · T2 · T3 · T4 · T6 · T5 · T7/T8.

**PRIORITY 2 (nach ersten Nutzerdaten):**
- FAQ-Section (existiert nicht): „Ist das Anlageberatung?", „Woher die Daten?", „Unterschied zu
  ChatGPT?", „Was heißt eine Analyse-Einheit?".
- Indirekte ChatGPT-Abgrenzung: „Eine KI gibt dir eine Antwort. ComAnalysis gibt dir den Ort, an
  dem du selbst analysierst und deine Arbeit behältst." (ohne Anti-KI-Ton).
- Erste echte Produkt-Screenshots (Ergebnis mit `SourceBadge`, Custom-Builder, Compare).

**PRIORITY 3 (Vision – noch NICHT bewerben):** Custom Analysis Frameworks, Investment-Thesis-
Tracking, KI-Research-Assistent, qualitative + quantitative Frameworks, historische
Analyse-Versionen, „Investment Research Workspace / Operating System".

---

## Verifikation

1. `cd frontend && npm run build` — **echter Build** (nicht nur `tsc --noEmit`), muss grün sein.
2. Dev-Server starten, `/` (bzw. `/landing`) + `/pricing` prüfen:
   - Hero-Copy neu; 3 Zeilen sauber; Gradient-Zeile ok; CTAs klickbar.
   - Neue Sections „Eigene Analyse" & „Zahlen verstehen" gerendert; mobil kein H-Scroll.
   - „7 Methoden" überall konsistent, nirgends „8"/„Acht".
   - Keine KI-Claims; keine neuen Yahoo/Alpha-Vantage-Textänderungen.
3. Responsive-Check (mobil/desktop, dark/light) für die zwei neuen Sections.
4. Deutsch, Ton konsistent; Neutralitäts-Leitplanke „keine Anlageberatung" bleibt erhalten.
