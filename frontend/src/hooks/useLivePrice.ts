import { useEffect, useState } from "react";
import { getCurrentPrice } from "../api/marketData";

const DEFAULT_POLL_INTERVAL_MS = 20000;

/** Polls the live-price endpoint for a single symbol, pausing while the tab
 * is hidden so backgrounded dashboards/sidebars don't keep hammering the
 * (rate-limited, yfinance-backed) endpoint. Used by LivePriceBadge wherever
 * a symbol is shown — sidebar favorites, dashboard history, analyze results. */
export function useLivePrice(symbol: string | null | undefined, intervalMs = DEFAULT_POLL_INTERVAL_MS) {
  const [price, setPrice] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(Boolean(symbol));

  useEffect(() => {
    const normalizedSymbol = symbol?.trim().toUpperCase();

    if (!normalizedSymbol) {
      setPrice(null);
      setError(null);
      setIsLoading(false);
      return;
    }

    const symbolToPoll: string = normalizedSymbol;
    let isCancelled = false;
    setIsLoading(true);

    async function fetchPrice() {
      try {
        const response = await getCurrentPrice(symbolToPoll);
        if (isCancelled) return;

        if (response.error || response.price == null) {
          setError(response.error ?? "Preis nicht verfügbar");
        } else {
          setPrice(response.price);
          setError(null);
        }
      } catch {
        if (!isCancelled) {
          setError("Preis nicht verfügbar");
        }
      } finally {
        if (!isCancelled) {
          setIsLoading(false);
        }
      }
    }

    fetchPrice();

    function handleVisibilityChange() {
      if (document.visibilityState === "visible") {
        fetchPrice();
      }
    }

    document.addEventListener("visibilitychange", handleVisibilityChange);

    const interval = window.setInterval(() => {
      if (document.visibilityState === "visible") {
        fetchPrice();
      }
    }, intervalMs);

    return () => {
      isCancelled = true;
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [symbol, intervalMs]);

  return { price, error, isLoading };
}
