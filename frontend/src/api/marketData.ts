import { apiRequest } from "./client";

export type CurrentPriceResponse = {
  symbol: string;
  price?: number;
  fetched_at?: string;
  error?: string;
};

export async function getCurrentPrice(symbol: string): Promise<CurrentPriceResponse> {
  return apiRequest<CurrentPriceResponse>(
    `/metrics/current-price/${encodeURIComponent(symbol)}`
  );
}
