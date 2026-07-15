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
  // EV-013: statt den Backend-Fehler still hinter dem 23-Symbol-Fallback zu
  // verstecken, machen wir ihn hier sichtbar - Aufrufer zeigen dann einen
  // Hinweis über der (weiterhin nutzbaren) Fallback-Liste.
  const [isDegraded, setIsDegraded] = useState(false);

  useEffect(() => {
    let isCancelled = false;
    // Klassisches Loading-Flag vor einem (hier debounced) Fetch - legitimer
    // Effect-Zweck laut React-Doku ("Fetching data"), kein Fall von
    // Zustand, der stattdessen während des Renders berechnet werden könnte
    // (LAUNCH_AUDIT.md P2-10).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setIsLoadingSuggestions(true);

    const timeoutId = window.setTimeout(() => {
      searchSymbolsSafe(query, limit)
        .then(({ entries, degraded }) => {
          if (isCancelled) return;
          setSuggestions(entries);
          setIsDegraded(degraded);
        })
        .catch(() => {
          if (isCancelled) return;
          setSuggestions([]);
          setIsDegraded(true);
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

  return { suggestions, isLoadingSuggestions, isDegraded };
}
