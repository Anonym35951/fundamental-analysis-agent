import { getCurrentPrice } from "../api/marketData";

const POLL_INTERVAL_MS = 20000;

export type PriceState = {
  price: number | null;
  error: string | null;
  isLoading: boolean;
};

type Listener = () => void;

type Entry = {
  state: PriceState;
  listeners: Set<Listener>;
  lastFetchedAt: number;
  inFlight: Promise<void> | null;
};

const INITIAL_STATE: PriceState = { price: null, error: null, isLoading: true };

// EV-113: EIN modul-globaler Store statt eines eigenen 20s-setInterval pro
// LivePriceBadge-Instanz (frueher useLivePrice.ts) - Sidebar-Favoriten und
// Dashboard-Sektion zeigen oft dieselben Symbole gleichzeitig, was vorher zu
// ~2x so vielen /current-price-Requests fuehrte wie es eindeutige Symbole
// gibt. Ref-Counting per Listener-Set: ein Symbol wird nur gepollt, solange
// mindestens ein Badge es abonniert hat.
const entries = new Map<string, Entry>();
let tickTimer: number | null = null;
let visibilityListenerAttached = false;

function ensureEntry(symbol: string): Entry {
  let entry = entries.get(symbol);
  if (!entry) {
    entry = { state: INITIAL_STATE, listeners: new Set(), lastFetchedAt: 0, inFlight: null };
    entries.set(symbol, entry);
  }
  return entry;
}

function notify(symbol: string): void {
  const entry = entries.get(symbol);
  if (!entry) return;
  for (const listener of entry.listeners) listener();
}

function isFresh(entry: Entry): boolean {
  return entry.lastFetchedAt > 0 && Date.now() - entry.lastFetchedAt < POLL_INTERVAL_MS;
}

async function fetchPrice(symbol: string): Promise<void> {
  const entry = entries.get(symbol);
  if (!entry || entry.inFlight) return;

  const promise = (async () => {
    try {
      const response = await getCurrentPrice(symbol);
      const current = entries.get(symbol);
      if (!current) return; // vollstaendig abbestellt, waehrend der Request lief

      if (response.error || response.price == null) {
        current.state = { price: current.state.price, error: response.error ?? "Preis nicht verfügbar", isLoading: false };
      } else {
        current.state = { price: response.price, error: null, isLoading: false };
      }
      current.lastFetchedAt = Date.now();
      notify(symbol);
    } catch {
      const current = entries.get(symbol);
      if (!current) return;
      current.state = { price: current.state.price, error: "Preis nicht verfügbar", isLoading: false };
      current.lastFetchedAt = Date.now();
      notify(symbol);
    } finally {
      const current = entries.get(symbol);
      if (current) current.inFlight = null;
    }
  })();

  entry.inFlight = promise;
  await promise;
}

function pollStaleVisibleSymbols(): void {
  if (document.visibilityState !== "visible") return;
  for (const [symbol, entry] of entries) {
    if (entry.listeners.size === 0) continue;
    if (!isFresh(entry)) fetchPrice(symbol);
  }
}

function ensureGlobalTick(): void {
  if (tickTimer === null) {
    tickTimer = window.setInterval(pollStaleVisibleSymbols, POLL_INTERVAL_MS);
  }
  if (!visibilityListenerAttached) {
    visibilityListenerAttached = true;
    document.addEventListener("visibilitychange", pollStaleVisibleSymbols);
  }
}

function stopGlobalTickIfIdle(): void {
  if (entries.size > 0) return;
  if (tickTimer !== null) {
    window.clearInterval(tickTimer);
    tickTimer = null;
  }
  if (visibilityListenerAttached) {
    document.removeEventListener("visibilitychange", pollStaleVisibleSymbols);
    visibilityListenerAttached = false;
  }
}

/** Abonniert Preis-Updates fuer `symbol` (bereits normalisiert erwartet).
 * Loest bei der ersten Subscription eines Symbols sofort einen Fetch aus
 * (sofern nicht bereits < 20s alt), danach uebernimmt der geteilte Tick.
 * Gibt eine Unsubscribe-Funktion zurueck; sobald der letzte Listener eines
 * Symbols abbestellt, wird dessen Eintrag entfernt (kein Leak, kein
 * Weiter-Pollen vergessener Symbole). */
export function subscribePrice(symbol: string, listener: Listener): () => void {
  const entry = ensureEntry(symbol);
  entry.listeners.add(listener);

  if (!isFresh(entry)) {
    fetchPrice(symbol);
  }

  ensureGlobalTick();

  return () => {
    const current = entries.get(symbol);
    if (!current) return;
    current.listeners.delete(listener);
    if (current.listeners.size === 0) {
      entries.delete(symbol);
      stopGlobalTickIfIdle();
    }
  };
}

/** Synchroner Snapshot fuer useSyncExternalStore - liefert den letzten
 * bekannten Stand sofort (kein "…"-Flackern bei einem Remount, solange
 * irgendein anderer Verbraucher dasselbe Symbol noch offen haelt). */
export function getPriceSnapshot(symbol: string): PriceState {
  return entries.get(symbol)?.state ?? INITIAL_STATE;
}
