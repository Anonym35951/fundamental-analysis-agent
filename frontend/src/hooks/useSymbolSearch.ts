import { useEffect, useState } from "react";
import { searchSymbolsSafe, type SymbolMeta } from "../api/analysis";

const DEBOUNCE_MS = 250;

/** Server-seitige Symbol-Suche (query-as-you-type) mit Debounce - ersetzt
 * das frühere "alle ~23 Symbole einmal laden, dann client-seitig filtern",
 * das mit dem vollen NYSE+NASDAQ-Universum (~6-7k Zeilen) nicht mehr
 * praktikabel ist (siehe api/routes/analyze.py: search_symbols). Ein leerer
 * Query liefert serverseitig eine kuratierte Popular-Liste, verhält sich
 * also wie das bisherige "Fokus ohne Eingabe zeigt Vorschläge"-Verhalten. */
export function useSymbolSearch(query: string, limit = 8) {
  const [suggestions, setSuggestions] = useState<SymbolMeta[]>([]);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(true);

  useEffect(() => {
    let isCancelled = false;
    setIsLoadingSuggestions(true);

    const timeoutId = window.setTimeout(() => {
      searchSymbolsSafe(query, limit)
        .then((result) => {
          if (!isCancelled) setSuggestions(result);
        })
        .catch(() => {
          if (!isCancelled) setSuggestions([]);
        })
        .finally(() => {
          if (!isCancelled) setIsLoadingSuggestions(false);
        });
    }, DEBOUNCE_MS);

    return () => {
      isCancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [query, limit]);

  return { suggestions, isLoadingSuggestions };
}
