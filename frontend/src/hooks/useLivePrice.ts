import { useCallback, useSyncExternalStore } from "react";
import { getPriceSnapshot, subscribePrice, type PriceState } from "./priceStore";

const EMPTY_STATE: PriceState = { price: null, error: null, isLoading: false };

function noopSubscribe(): () => void {
  return () => {};
}

/** Liest den Live-Preis eines Symbols aus dem geteilten `priceStore`
 * (EV-113) statt selbst zu pollen — mehrere gleichzeitige Aufrufer desselben
 * Symbols (z. B. Sidebar-Favorit + Dashboard-Kachel) teilen sich einen
 * einzigen 20s-Tick statt je einen eigenen `setInterval` zu starten. Rückgabe-
 * Signatur unveraendert gegenueber der vorherigen Eigenimplementierung, damit
 * LivePriceBadge und alle anderen Aufrufer unangetastet bleiben. */
export function useLivePrice(symbol: string | null | undefined) {
  const normalizedSymbol = symbol?.trim().toUpperCase() || null;

  const subscribe = useCallback(
    (onStoreChange: () => void) => {
      if (!normalizedSymbol) return noopSubscribe();
      return subscribePrice(normalizedSymbol, onStoreChange);
    },
    [normalizedSymbol]
  );

  const getSnapshot = useCallback(() => {
    if (!normalizedSymbol) return EMPTY_STATE;
    return getPriceSnapshot(normalizedSymbol);
  }, [normalizedSymbol]);

  return useSyncExternalStore(subscribe, getSnapshot);
}
