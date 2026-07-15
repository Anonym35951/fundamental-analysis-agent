import { getMetricConfig, normalizeMetricKey } from "../../config/metricsConfig";

export type CriterionLike = {
  operator?: ">" | "<" | ">=" | "<=";
  threshold?: number;
};

/** Centralizes the value/label formatting logic that used to live duplicated
 * inline across the analysis result views. Both standard and custom
 * analysis results render through these helpers so formatting stays
 * consistent. */

export function formatLabel(key: string): string {
  const config = getMetricConfig(key);

  if (config?.label) {
    return config.label;
  }

  return normalizeMetricKey(key)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function formatCompactNumber(value: number): string {
  const abs = Math.abs(value);

  if (abs >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2).replace(/\.?0+$/, "")}B`;
  }
  if (abs >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2).replace(/\.?0+$/, "")}M`;
  }
  if (abs >= 1_000) {
    return `${(value / 1_000).toFixed(2).replace(/\.?0+$/, "")}K`;
  }

  return value.toFixed(2).replace(/\.?0+$/, "");
}

/** ISO-Code → Symbol für die vier vom Feedback konkret genannten Währungen
 * (EVOLVING.md EV-022). Unbekannte Codes werden als Suffix angehängt statt
 * geraten - lieber "1,2 Mrd. CNY" als ein falsches Symbol. */
const CURRENCY_SYMBOLS: Record<string, string> = {
  USD: "$",
  EUR: "€",
  GBP: "£",
  JPY: "¥",
};

/** Zentrale Geldbetrag-Formatierung (EVOLVING.md EV-022) - ersetzt das
 * vorher hartcodierte `$`. Ohne `isoCode` (unbekannte/nicht ermittelte
 * Berichtswährung) bleibt das bisherige Verhalten exakt erhalten: `$` als
 * Präfix, keine sichtbare Änderung für bestehende Analysen. */
export function formatMonetary(value: number, isoCode?: string | null): string {
  const compact = formatCompactNumber(value);

  if (!isoCode) {
    return `$${compact}`;
  }

  const symbol = CURRENCY_SYMBOLS[isoCode];
  if (symbol) {
    return `${symbol}${compact}`;
  }

  return `${compact} ${isoCode}`;
}

function isDollarKey(key: string): boolean {
  // Ratio/multiple metrics (price_to_ebit, cash_to_market_cap, ev_to_sales,
  // debt_to_equity, ...) are unitless and never a currency amount, even
  // though their key often contains "price"/"cash"/etc. — check this first
  // so the substring heuristic below doesn't misfire on them.
  if (key.includes("_to_")) return false;

  return (
    key.includes("price") ||
    key.includes("cash") ||
    key.includes("market_cap") ||
    key.includes("target") ||
    key.includes("upside") ||
    key.includes("downside") ||
    key.includes("assets") ||
    key.includes("liabilities") ||
    key.includes("book") ||
    key.includes("tbv")
  );
}

export function formatMetricValue(value: unknown, key?: string, currency?: string | null): string {
  if (value === null || value === undefined) return "—";

  const normalizedKey = key ? normalizeMetricKey(key) : "";
  const config = key ? getMetricConfig(key) : undefined;

  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);

    const decimals = config?.decimals ?? 2;
    const scaledValue = value * (config?.scale ?? 1);
    const formatted = scaledValue.toFixed(decimals).replace(/\.?0+$/, "");

    if (config?.unit === "%") {
      return `${formatted}%`;
    }
    if (config?.unit === "currency" || isDollarKey(normalizedKey)) {
      return formatMonetary(value, currency);
    }

    return formatted;
  }

  if (typeof value === "boolean") {
    return value ? "Ja" : "Nein";
  }

  if (Array.isArray(value)) {
    return value.map((item) => formatMetricValue(item, key, currency)).join(", ");
  }

  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([objectKey, objectValue]) => `${formatLabel(objectKey)}: ${formatMetricValue(objectValue, objectKey, currency)}`)
      .join(" · ");
  }

  return String(value);
}

/** Evaluates a simple threshold criterion (operator + threshold) against a
 * numeric value, mirroring the pass/fail semantics already used by the
 * hardcoded AgentAction analysis modes, but data-driven for Custom Analysis. */
export function isPassFail(criterion: CriterionLike | null | undefined, value: unknown): boolean | null {
  if (!criterion?.operator || typeof criterion.threshold !== "number") return null;
  if (typeof value !== "number" || !Number.isFinite(value)) return null;

  switch (criterion.operator) {
    case ">":
      return value > criterion.threshold;
    case "<":
      return value < criterion.threshold;
    case ">=":
      return value >= criterion.threshold;
    case "<=":
      return value <= criterion.threshold;
    default:
      return null;
  }
}

const highlightStyle = {
  background: "rgba(250, 204, 21, 0.22)",
  color: "#fef9c3",
  borderRadius: "5px",
  padding: "0 3px",
};

export function highlightText(text: string, query?: string) {
  if (!query?.trim()) return text;

  const safeQuery = query.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const parts = text.split(new RegExp(`(${safeQuery})`, "gi"));

  return (
    <>
      {parts.map((part, index) =>
        part.toLowerCase() === query.trim().toLowerCase() ? (
          <mark key={index} style={highlightStyle}>
            {part}
          </mark>
        ) : (
          part
        )
      )}
    </>
  );
}
