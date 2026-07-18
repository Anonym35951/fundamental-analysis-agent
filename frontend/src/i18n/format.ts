import { INTL_LOCALES, type Locale } from "./config";
import type { PercentChangeResult } from "../components/charts/chartUtils";

/** EVOLVING.md § 9: zentrale, locale-aware Formatter. DE-Zweige sind
 * wörtliche Kopien der bisherigen Implementierungen aus
 * `components/charts/chartUtils.ts` / `components/metrics/metricFormatting.tsx`
 * — abgesichert durch die Characterization-Tests in
 * `i18n/format.characterization.test.ts` (I18N-001). Beide Module
 * delegieren hierher (I18N-003), rufen aber vorerst mit fest verdrahtetem
 * `"de"` auf — kein Verhalten ändert sich in Phase 1. Der `locale`-Parameter
 * wird erst in späteren Phasen (Analysis, Dashboard, ...) tatsächlich
 * durchgereicht. */

const CHART_COMPACT_SUFFIX: Record<Locale, { thousand: string; million: string; billion: string }> = {
  de: { thousand: "Tsd.", million: "Mio.", billion: "Mrd." },
  en: { thousand: "K", million: "M", billion: "B" },
};

/** Chart-Achsen/Tooltip-Kompaktformat (§ 9.1). DE-Zweig = heutige
 * chartUtils.formatCompactNumber-Logik, identische Schwellen/Rundung. */
export function formatCompactNumberChart(value: number, locale: Locale): string {
  const abs = Math.abs(value);
  const { thousand, million, billion } = CHART_COMPACT_SUFFIX[locale];

  if (abs >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(abs >= 10_000_000_000 ? 0 : 1)} ${billion}`;
  }
  if (abs >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1)} ${million}`;
  }
  if (abs >= 1_000) {
    return `${(value / 1_000).toFixed(abs >= 10_000 ? 0 : 1)} ${thousand}`;
  }
  return value.toLocaleString(INTL_LOCALES[locale], { maximumFractionDigits: 2 });
}

/** Metrik-Kompaktformat (§ 9.1). Bewusst NICHT locale-abhängig: bleibt für
 * DE unverändert K/M/B (Paritätsgebot schlägt Konsistenzwunsch — die
 * bestehende Inkonsistenz zu formatCompactNumberChart wird nicht
 * "mitrepariert", siehe EVOLVING.md § 9.1). */
export function formatCompactNumberMetric(value: number): string {
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

/** Prozent-Veränderungs-Badge (§ 9.2). DE-Zweig = heutige
 * chartUtils.formatPercentChange-Logik (Komma, `±0,0 %`, `n. v.`). */
export function formatPercentChangeChart(result: PercentChangeResult, locale: Locale): string {
  const notAvailable = locale === "de" ? "n. v." : "n/a";
  if (result.percent === null) return notAvailable;

  const rounded = Math.round(result.percent * 10) / 10;
  const zero = locale === "de" ? "±0,0 %" : "±0.0%";
  if (rounded === 0) return zero;

  const sign = rounded > 0 ? "+" : "";
  const formatted = rounded.toLocaleString(INTL_LOCALES[locale], { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  return locale === "de" ? `${sign}${formatted} %` : `${sign}${formatted}%`;
}

export function formatNumber(value: number, locale: Locale, opts?: Intl.NumberFormatOptions): string {
  return value.toLocaleString(INTL_LOCALES[locale], opts);
}

export function formatPercent(value: number, locale: Locale, decimals = 1): string {
  const formatted = value.toLocaleString(INTL_LOCALES[locale], {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return locale === "de" ? `${formatted} %` : `${formatted}%`;
}

/** DE-Zweig entspricht dem heute im Code häufigsten Muster
 * (`toLocaleDateString("de-DE", {day:"2-digit", month:"2-digit", year:"numeric"})`,
 * u. a. AccountPage.tsx, SourceBadge.tsx). Call-Site-Migration erfolgt
 * phasenweise (§ 22), nicht in Phase 1. */
export function formatDate(iso: string | Date, locale: Locale, style: "short" | "medium" = "short"): string {
  const date = typeof iso === "string" ? new Date(iso) : iso;
  const options: Intl.DateTimeFormatOptions =
    style === "short"
      ? { day: "2-digit", month: "2-digit", year: "numeric" }
      : { dateStyle: "medium", timeStyle: "short" };
  return date.toLocaleDateString(INTL_LOCALES[locale], options);
}

const CURRENCY_SYMBOLS: Record<string, string> = { USD: "$", EUR: "€", GBP: "£", JPY: "¥" };

/** Währung kommt ausschließlich als ISO-Code vom Backend (`reporting_currency`
 * / `currency`) — Locale ändert NIEMALS die Währung, nur die Darstellung
 * (§ 9.4). */
export function formatCurrency(value: number, currencyIso: string | null | undefined, locale: Locale): string {
  const compact = formatCompactNumberMetric(value);
  if (!currencyIso) return `$${compact}`;

  const symbol = CURRENCY_SYMBOLS[currencyIso];
  if (symbol) return `${symbol}${compact}`;

  return locale === "de" ? `${compact} ${currencyIso}` : `${compact} ${currencyIso}`;
}

export function formatBoolean(value: boolean, locale: Locale): string {
  if (locale === "de") return value ? "Ja" : "Nein";
  return value ? "Yes" : "No";
}
