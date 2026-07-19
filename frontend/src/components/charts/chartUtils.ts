export type ChartLayer = {
  id: string;
  label: string;
  data: Array<{ date: string; value: number }>;
  axis: "left" | "right";
  color: string;
  /** ISO-Währungscode dieser Firma für diese Kennzahl (EVOLVING.md EV-023).
   * Nur für `unit==="currency"`-Kennzahlen gesetzt - Ratio-/Margen-Charts
   * lassen dies bewusst `undefined`, damit sie nie eine Währungskennzeichnung
   * bekommen. `null`/fehlend bedeutet "unbekannt", nicht "keine Währung". */
  currency?: string | null;
};

/** EVOLVING.md EV-030: wie Zeitpunkte auf der X-Achse zusammengefasst
 * werden, bevor mehrere Firmen-Layer zu gemeinsamen Zeilen gemergt werden.
 * "year"/"quarter" ordnen den KALENDERjahres-/-quartalswert des Datums zu,
 * nicht ein Geschäftsjahr - bewusste, dokumentierte Vereinfachung: ohne
 * Fiskaljahres-Metadaten (z. B. "Geschäftsjahresende im Juni") wäre jede
 * Zuordnung eines Jan-Jun-Stichtags zum Vor- oder laufenden Jahr reine
 * Spekulation. "date" ist ein reiner Durchreicher (kein Bucketing) - für
 * Einzelfirmen-Charts und künftige tägliche Kurscharts. */
export type BucketMode = "year" | "quarter" | "date";

/** Reduziert ein "YYYY-MM-DD"-Datum auf einen groeberen Perioden-Bucket,
 * damit mehrere Firmen mit unterschiedlichen Fiskal-Stichtagen im selben
 * Kalenderjahr/-quartal auf demselben Merge-Schluessel landen (Root Cause
 * des Tooltip-Bugs, EVOLVING.md Abschnitt 5.3/6 P5). */
export function bucketKey(dateStr: string, mode: BucketMode): string {
  if (mode === "date") return dateStr;

  const [yearStr, monthStr] = dateStr.split("-");
  if (mode === "year") return yearStr;

  const month = Number(monthStr);
  const quarter = Math.max(1, Math.min(4, Math.ceil(month / 3) || 1));
  return `Q${quarter} ${yearStr}`;
}

export type MergedChartRow = Record<string, string | number> & { date: string; label: string };

/** Merged mehrere Firmen-Layer zu einer gemeinsamen Zeile pro Bucket
 * (EVOLVING.md EV-030) - vorher wurde nach dem EXAKTEN Datums-String
 * geschluesselt, wodurch Firmen mit abweichenden Fiskal-Stichtagen
 * (z. B. 30.09. vs. 31.12.) praktisch nie in derselben Zeile landeten.
 *
 * Pro Layer+Bucket gewinnt der Punkt mit dem spaetesten Original-Datum
 * (z. B. bei einem Geschaeftsjahreswechsel innerhalb desselben Buckets).
 * Die Zeile selbst speichert zusaetzlich `date` (das spaeteste in diesem
 * Bucket beobachtete Original-Datum, über alle Layer) fuer eine korrekte
 * chronologische Sortierung - ein reiner String-Vergleich der Bucket-Labels
 * waere hier falsch ("Q4 2023" < "Q1 2024" alphabetisch nicht wahr) - und
 * `label` (der Bucket-Text) fuer Achse und Tooltip. */
export function mergeLayers(layers: ChartLayer[], mode: BucketMode = "date"): MergedChartRow[] {
  const buckets = new Map<string, { rowDate: string; cellDates: Map<string, string>; values: Record<string, number> }>();

  for (const layer of layers) {
    for (const point of layer.data) {
      const bucket = bucketKey(point.date, mode);
      let entry = buckets.get(bucket);
      if (!entry) {
        entry = { rowDate: point.date, cellDates: new Map(), values: {} };
        buckets.set(bucket, entry);
      }

      const existingCellDate = entry.cellDates.get(layer.id);
      if (existingCellDate === undefined || point.date > existingCellDate) {
        entry.cellDates.set(layer.id, point.date);
        entry.values[layer.id] = point.value;
      }
      if (point.date > entry.rowDate) {
        entry.rowDate = point.date;
      }
    }
  }

  return Array.from(buckets.entries())
    .map(([bucket, entry]) => ({ date: entry.rowDate, label: bucket, ...entry.values }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

/** Deliberate exception to the app's otherwise-monochrome theme: the compare
 * chart can overlay many company×metric series at once, so it needs a large,
 * highly saturated, mutually-distinguishable categorical palette instead of
 * the 6-shade greyscale ramp used elsewhere. Colors alternate warm/cool/hue
 * families so consecutive indices (sequential assignment in
 * compare/mapping.ts) never land next to a visually similar neighbor. Cycles
 * only once a comparison exceeds this many series. */
export const LAYER_COLORS = [
  "#22d3ee", // cyan
  "#fb7185", // rose
  "#a3e635", // lime
  "#c084fc", // violet
  "#fbbf24", // amber
  "#38bdf8", // sky
  "#f472b6", // pink
  "#4ade80", // green
  "#fb923c", // orange
  "#818cf8", // indigo
  "#facc15", // yellow
  "#2dd4bf", // teal
  "#f87171", // red
  "#a78bfa", // purple
  "#34d399", // emerald
  "#e879f9", // fuchsia
  "#60a5fa", // blue
  "#fde047", // pale yellow
  "#d946ef", // magenta
  "#5eead4", // light teal
];

/** Compact German number formatting (k / Mio. / Mrd.) so large axis ticks
 * and tooltip values (market cap, enterprise value, ...) stay readable
 * instead of rendering as long unbroken digit strings. */
export function formatCompactNumber(value: number): string {
  const abs = Math.abs(value);

  if (abs >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(abs >= 10_000_000_000 ? 0 : 1)} Mrd.`;
  }
  if (abs >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1)} Mio.`;
  }
  if (abs >= 1_000) {
    return `${(value / 1_000).toFixed(abs >= 10_000 ? 0 : 1)} Tsd.`;
  }
  return value.toLocaleString("de-DE", { maximumFractionDigits: 2 });
}

export type TooltipRow = {
  id: string;
  label: string;
  color: string;
  /** `null`, wenn dieser Layer im gehoverten Bucket keinen Wert hat -
   * Aufrufer zeigen dafür "–" statt die Firma wegzulassen (EVOLVING.md
   * EV-031, D2). */
  value: number | null;
  currency?: string | null;
};

/** Reine Wert-Extraktionslogik für den Custom-Tooltip (EVOLVING.md EV-031):
 * iteriert über ALLE `layers` (nicht nur die im gemergten Chart-Row
 * tatsächlich vorhandenen Keys), damit jede ausgewählte Firma an jeder
 * Hover-Position auftaucht, auch ohne Wert im gehoverten Bucket. Als reine
 * Funktion (kein React) unabhängig von der Recharts-Tooltip-Komponente
 * testbar. */
export function extractTooltipRows(
  row: Record<string, string | number> | undefined,
  layers: ChartLayer[]
): TooltipRow[] {
  if (!row) return [];

  return layers.map((layer) => {
    const rawValue = row[layer.id];
    return {
      id: layer.id,
      label: layer.label,
      color: layer.color,
      value: typeof rawValue === "number" ? rawValue : null,
      currency: layer.currency,
    };
  });
}

/** Leitet aus den Currency-Codes aller Layer eines Charts ab, ob eine
 * einheitliche Währung angezeigt werden kann, mehrere Originalwährungen
 * gemischt vorliegen (Hinweis-Badge nötig), oder gar keine Währung bekannt
 * ist (z. B. Ratio-/Margen-Charts, die `currency` bewusst nie setzen -
 * EVOLVING.md EV-023). */
export type ChartCurrencyState = { uniform: string } | { mixed: string[] } | { none: true };

export function layersCurrencyState(layers: ChartLayer[]): ChartCurrencyState {
  const codes = Array.from(new Set(layers.map((layer) => layer.currency).filter((code): code is string => Boolean(code))));

  if (codes.length === 0) return { none: true };
  if (codes.length === 1) return { uniform: codes[0] };
  return { mixed: codes };
}

/** EVOLVING.md EV-040: verfügbare Zeitraumfenster für den Chart-Zeitraumfilter.
 * "max" ist bewusst kein Eintrag der Monats-Map - er bedeutet "kein Cutoff",
 * nicht "sehr viele Monate zurückrechnen". */
export type TimeRange = "1m" | "2m" | "3m" | "6m" | "1y" | "2y" | "5y" | "max";

export const TIME_RANGE_MONTHS: Record<Exclude<TimeRange, "max">, number> = {
  "1m": 1,
  "2m": 2,
  "3m": 3,
  "6m": 6,
  "1y": 12,
  "2y": 24,
  "5y": 60,
};

export const TIME_RANGE_OPTIONS: Array<{ value: TimeRange; label: string }> = [
  { value: "1m", label: "1M" },
  { value: "2m", label: "2M" },
  { value: "3m", label: "3M" },
  { value: "6m", label: "6M" },
  { value: "1y", label: "1J" },
  { value: "2y", label: "2J" },
  { value: "5y", label: "5J" },
  { value: "max", label: "Max" },
];

/** Zieht `months` Kalendermonate von einem "YYYY-MM-DD"-Datum ab, mit
 * Monatsends-Klemmung (z. B. 31.03. − 1 Monat -> 28./29.02., je nach
 * Schaltjahr) statt eines naiven `Date`-Rollovers (der 31.03. − 1 Monat
 * sonst stillschweigend auf den 03.03. verschieben würde). Keine neue
 * Dependency (EVOLVING.md EV-040). */
function subtractMonths(dateStr: string, months: number): string {
  const [year, month, day] = dateStr.split("-").map(Number);

  const totalMonths = year * 12 + (month - 1) - months;
  const targetYear = Math.floor(totalMonths / 12);
  const targetMonthIndex = totalMonths - targetYear * 12; // 0-basiert

  const daysInTargetMonth = new Date(targetYear, targetMonthIndex + 1, 0).getDate();
  const clampedDay = Math.min(day, daysInTargetMonth);

  const mm = String(targetMonthIndex + 1).padStart(2, "0");
  const dd = String(clampedDay).padStart(2, "0");
  return `${targetYear}-${mm}-${dd}`;
}

/** Reine Zeitraumfilter-Funktion (EVOLVING.md EV-040): Anker ist das NEUESTE
 * Datum der übergebenen Serie (nicht "heute"), damit ein Chart bei
 * Datenverzug (z. B. Fundamentaldaten sind Monate alt) nicht plötzlich leer
 * erscheint. `anchorDate` kann explizit übergeben werden, damit mehrere
 * Firmen-Layer eines Charts synchron am selben Kalenderdatum geschnitten
 * werden (EV-041), statt jede Serie an ihrem eigenen letzten Punkt. */
export function filterSeriesByRange(
  series: Array<{ date: string; value: number }>,
  range: TimeRange,
  anchorDate?: string
): Array<{ date: string; value: number }> {
  if (range === "max" || series.length === 0) return series;

  const anchor = anchorDate ?? series.reduce((latest, point) => (point.date > latest ? point.date : latest), series[0].date);
  const cutoff = subtractMonths(anchor, TIME_RANGE_MONTHS[range]);

  return series.filter((point) => point.date >= cutoff);
}

/** Filtert ALLE Layer eines Charts auf denselben Zeitraum mit einem
 * GEMEINSAMEN Anker (dem neuesten Datum über alle Layer hinweg), statt jede
 * Firmen-Serie an ihrem eigenen letzten Punkt zu kappen - sonst würden
 * Firmen mit leicht versetzten Fiskal-/Handelstagen bei jedem Rangewechsel
 * wieder auseinanderdriften (EVOLVING.md EV-041, aufbauend auf dem
 * Bucketing-Fix aus EV-030). Generisch über `T`, damit sowohl das schlanke
 * `ChartLayer` (chartUtils/MultiLayerChart) als auch das reichhaltigere
 * `CompareLayer` (inkl. Metadaten wie `groupId`/`currency`) unverändert
 * durchgereicht werden können. */
export function filterChartLayers<T extends { data: Array<{ date: string; value: number }> }>(
  layers: T[],
  range: TimeRange
): T[] {
  if (range === "max") return layers;

  const anchor = layers
    .flatMap((layer) => layer.data)
    .reduce((latest, point) => (point.date > latest ? point.date : latest), "");

  if (!anchor) return layers;

  return layers.map((layer) => ({ ...layer, data: filterSeriesByRange(layer.data, range, anchor) }));
}

/** EVOLVING.md EV-050: Ergebnis von `computePercentChange` - entweder ein
 * berechenbarer Prozentwert, oder `null` mit einem Grund, warum keine
 * sinnvolle Zahl gebildet werden kann. Der Grund ist Teil des Vertrags
 * (nicht nur `null`), damit die UI (EV-051) unterscheiden kann zwischen
 * "zu wenig Daten" und "Startwert 0/negativ" (mathematisch irreführend,
 * nicht bloß fehlend). */
export type PercentChangeResult = { percent: number } | { percent: null; reason: "insufficient" | "zero-start" | "negative-start" };

/** Reine %-Veränderungs-Berechnung (EVOLVING.md EV-050): Start = erster,
 * Ende = letzter GÜLTIGER (endlicher) Punkt der Serie - `NaN`/`Infinity` an
 * den Rändern (defensiv, sollte in der Praxis nicht vorkommen) werden nach
 * innen übersprungen statt die Serie fälschlich als "zu wenig Punkte"
 * abzulehnen. Erwartet eine bereits zeitraum-gefilterte Serie (EV-040) -
 * diese Funktion filtert selbst nicht nach Datum.
 *
 * WICHTIG: "erster"/"letzter" Punkt meint das früheste/späteste `date`,
 * NICHT Array-Index 0/length-1 - Backend-Serien sind nicht durchgängig
 * chronologisch aufsteigend sortiert (z. B. `sec_source.py` liefert manche
 * historischen Reihen absteigend, neuestes Datum zuerst). Ein naiver
 * Array-Positions-Zugriff hätte hier Start/Ende vertauscht und z. B. aus
 * jahrzehntelangem Umsatzwachstum einen scheinbaren Einbruch von -95 %
 * gemacht (live in EV-051 beobachtet und hier korrigiert). */
export function computePercentChange(series: Array<{ date: string; value: number }>): PercentChangeResult {
  let startPoint: { date: string; value: number } | null = null;
  let endPoint: { date: string; value: number } | null = null;
  let validCount = 0;

  for (const point of series) {
    if (!Number.isFinite(point.value)) continue;
    validCount++;
    if (!startPoint || point.date < startPoint.date) startPoint = point;
    if (!endPoint || point.date > endPoint.date) endPoint = point;
  }

  if (validCount < 2 || !startPoint || !endPoint) return { percent: null, reason: "insufficient" };

  const start = startPoint.value;
  const end = endPoint.value;

  if (start === 0) return { percent: null, reason: "zero-start" };
  if (start < 0) return { percent: null, reason: "negative-start" };

  return { percent: ((end - start) / start) * 100 };
}

/** Formatiert ein `PercentChangeResult` fürs UI (EVOLVING.md EV-050/051):
 * Vorzeichen, 1 Nachkommastelle, de-DE-Komma. Rundet auf 0,0 % gerundete
 * Werte (inkl. `-0`) auf ein neutrales „±0,0 %" statt eines irreführenden
 * „-0,0 %" oder eines nichtssagenden „+0,0 %". `null`-Ergebnisse (jeder
 * `reason`) werden einheitlich als „n. v." dargestellt - der Unterschied
 * zwischen den Gründen ist nur für Tooltips/Titel relevant, nicht für den
 * sichtbaren Badge-Text. */
export function formatPercentChange(result: PercentChangeResult): string {
  if (result.percent === null) return "n. v.";

  const rounded = Math.round(result.percent * 10) / 10;
  if (rounded === 0) return "±0,0 %";

  const sign = rounded > 0 ? "+" : "";
  const formatted = rounded.toLocaleString("de-DE", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  return `${sign}${formatted} %`;
}

/** EVOLVING.md EV-051: eine Kennzahl bekommt eine %-Veränderungs-Badge NUR,
 * wenn sie eine Level-Kennzahl in einer Währung ist (Umsatz, Market Cap,
 * Cashflows, Buchwerte, Kurs) - Ratios (EV/EBIT, P/B …), Margen/Prozent-
 * Kennzahlen und Pass/Fail-Werte bekommen bewusst keine Badge (derselbe
 * Geltungsbereich wie EV-050). Deckungsgleich mit der Currency-Kennzeichnung
 * aus EV-023 (`unit === "currency"`) - als eigenständige, aus `chartUtils`
 * heraus ohne Metrik-Katalog testbare Funktion, damit die Aufrufer
 * (ComparePage, CustomAnalysisResultsList) dieselbe Bedingung nicht separat
 * duplizieren. */
export function isPercentChangeEligibleUnit(unit: string | undefined): boolean {
  return unit === "currency";
}

/** EVOLVING.md EV-062: normalisiert eine Kursserie auf eine prozentuale
 * Veränderung ab dem frühesten Punkt der Serie ("Start = 0 %") - für den
 * normalisierten Kursvergleich mehrerer Firmen auf einer gemeinsamen
 * %-Skala (D4), statt absoluter Kurse in stark unterschiedlichen
 * Größenordnungen (z. B. eine 30-Dollar- neben einer 800-Dollar-Aktie).
 * Sucht wie `computePercentChange` (EV-050/051-Bugfix) den frühesten
 * GÜLTIGEN Punkt über das `date`-Feld, nicht über Array-Position - dieselbe
 * Absicherung gegen nicht-chronologisch sortierte Serien. Ein Startwert von
 * 0 ist bei Aktienkursen praktisch ausgeschlossen (anders als bei manchen
 * Fundamentalkennzahlen in EV-050); dieser Fall liefert defensiv eine leere
 * Serie statt einer Division durch 0. */
export function normalizePriceSeries(series: Array<{ date: string; value: number }>): Array<{ date: string; value: number }> {
  let basePoint: { date: string; value: number } | null = null;
  for (const point of series) {
    if (!Number.isFinite(point.value)) continue;
    if (!basePoint || point.date < basePoint.date) basePoint = point;
  }

  if (!basePoint || basePoint.value === 0) return [];

  const base = basePoint.value;
  return series
    .filter((point) => Number.isFinite(point.value))
    .map((point) => ({ date: point.date, value: ((point.value - base) / base) * 100 }));
}
