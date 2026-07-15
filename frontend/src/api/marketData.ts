import { apiRequest } from "./client";

export type CurrentPriceResponse = {
  symbol: string;
  price?: number;
  /** Immer "USD" (NYSE/NASDAQ-Symbol-Universum) - additiv seit EVOLVING.md
   * EV-021, kann bei sehr alten gecachten Responses fehlen. */
  currency?: string;
  fetched_at?: string;
  error?: string;
};

export async function getCurrentPrice(symbol: string): Promise<CurrentPriceResponse> {
  return apiRequest<CurrentPriceResponse>(
    `/metrics/current-price/${encodeURIComponent(symbol)}`
  );
}

/** EVOLVING.md EV-060: dieselben 8 Zeiträume wie das Frontend-Chart-Filter-
 * Pendant (`TimeRange` in `components/charts/chartUtils.ts`) - hier separat
 * deklariert statt importiert, damit die API-Schicht nicht von der
 * Komponenten-Schicht abhängt. */
export type PriceHistoryRange = "1m" | "2m" | "3m" | "6m" | "1y" | "2y" | "5y" | "max";

export type PriceHistoryRow = { date: string; close: number };

export type PriceHistoryResponse = {
  symbol: string;
  /** Immer "USD" (NYSE/NASDAQ-Symbol-Universum). */
  currency: string;
  range: PriceHistoryRange;
  rows: PriceHistoryRow[];
};

/** Tägliche (bei 2y/5y wöchentliche, bei sehr langer "max"-Historie
 * monatliche) Adjusted-Close-Kursreihe für Kurscharts (EVOLVING.md
 * EV-060/061/062). */
export async function getPriceHistory(symbol: string, range: PriceHistoryRange): Promise<PriceHistoryResponse> {
  return apiRequest<PriceHistoryResponse>(
    `/metrics/price-history/${encodeURIComponent(symbol)}?range=${encodeURIComponent(range)}`
  );
}

/** EVOLVING.md EV-070: nur die beiden kurzen Ranges - der Batch-Endpoint ist
 * für Sparklines gedacht, keine langfristigen Charts. */
export type PriceHistoryBatchRange = "1m" | "3m";

export type PriceHistoryBatchEntry = PriceHistoryResponse | { symbol: string; error: string };

export type PriceHistoryBatchResponse = { results: PriceHistoryBatchEntry[] };

/** Preishistorien für bis zu 20 Symbole in einem Request (EVOLVING.md
 * EV-070) - für die Dashboard-Favoriten-Sparklines (EV-071), damit N
 * Favoriten nicht N Einzel-Requests auslösen. Einzelne Symbole können im
 * Ergebnis-Array einen `error`-Eintrag statt `rows` haben (Teilfehler
 * blockieren nicht den ganzen Batch) - Aufrufer müssen beide Formen anhand
 * von `"rows" in entry` unterscheiden. */
export async function getPriceHistoryBatch(
  symbols: string[],
  range: PriceHistoryBatchRange
): Promise<PriceHistoryBatchResponse> {
  return apiRequest<PriceHistoryBatchResponse>(
    `/metrics/price-history-batch?symbols=${encodeURIComponent(symbols.join(","))}&range=${encodeURIComponent(range)}`
  );
}
