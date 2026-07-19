import { createContext } from "react";
import type { Locale } from "./config";
import type { Dictionary } from "./locales/de";
import type { MatchShape } from "./types";

// Laufzeit-Dictionaries sind gegen die DE-Struktur typisiert, aber mit
// generischen `string`-Blattwerten (MatchShape) statt DE-Literalen, weil
// dieselbe Variable je nach aktiver Locale entweder das DE- oder das
// EN-Dictionary enthält. Die literal-typisierte `Dictionary` (typeof de)
// wird nur für die Compile-Zeit-Key-Prüfung in useTranslation gebraucht,
// nicht für den tatsächlichen Laufzeit-Wert hier.
export type RuntimeDictionary = MatchShape<Dictionary>;

export type LocaleContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  /** true sobald das Dictionary der aktiven Locale geladen ist — für "de"
   * immer sofort true (eager), für "en" erst nach dem Lazy-Import. */
  ready: boolean;
  dictionary: RuntimeDictionary;
  /** DE dient als Fallback für einzelne fehlende EN-Keys (§ 15) — bei
   * vollständig typisierten Dictionaries im Normalfall nie gebraucht. */
  fallbackDictionary: RuntimeDictionary;
};

// Eigene Datei ohne Komponenten-Export, damit weder LocaleProvider.tsx
// (Provider) noch useLocale.ts (Hook) selbst einen Nicht-Komponenten-Wert
// exportieren (react-refresh/only-export-components) — gleiches Muster wie
// components/ui/toastContextValue.ts und themeModeContextValue.ts.
export const LocaleContext = createContext<LocaleContextValue | null>(null);
