import { apiRequest } from "./client";

export type DataSourceSummary = {
  symbol: string;
  frequency: string;
  source: string;
  /** ISO-Datum (YYYY-MM-DD) der jüngsten Berichtsperiode in den Rohdaten. */
  as_of: string | null;
  fetched_at: string;
};

export function getDataSourceSummary(symbol: string, frequency: "annual" | "quarterly" = "annual") {
  return apiRequest<DataSourceSummary>(
    `/metrics/data-source/${encodeURIComponent(symbol)}?frequency=${frequency}`
  );
}
