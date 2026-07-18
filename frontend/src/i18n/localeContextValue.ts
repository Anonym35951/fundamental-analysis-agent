import { createContext } from "react";
import type { Locale } from "./config";
import type { Dictionary } from "./locales/de";

export type LocaleContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  /** true sobald das Dictionary der aktiven Locale geladen ist — für "de"
   * immer sofort true (eager), für "en" erst nach dem Lazy-Import. */
  ready: boolean;
  dictionary: Dictionary;
  /** DE dient als Fallback für einzelne fehlende EN-Keys (§ 15) — bei
   * vollständig typisierten Dictionaries im Normalfall nie gebraucht. */
  fallbackDictionary: Dictionary;
};

// Eigene Datei ohne Komponenten-Export, damit weder LocaleProvider.tsx
// (Provider) noch useLocale.ts (Hook) selbst einen Nicht-Komponenten-Wert
// exportieren (react-refresh/only-export-components) — gleiches Muster wie
// components/ui/toastContextValue.ts und themeModeContextValue.ts.
export const LocaleContext = createContext<LocaleContextValue | null>(null);
