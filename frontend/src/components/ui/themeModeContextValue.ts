import { createContext } from "react";

export type ThemeMode = "light" | "dark";

export type ThemeModeContextValue = {
  mode: ThemeMode;
  toggleMode: () => void;
};

// Eigene Datei ohne Komponenten-Export, damit weder ThemeModeContext.tsx
// (Provider) noch useThemeMode.ts (Hook) selbst einen Nicht-Komponenten-Wert
// exportieren (react-refresh/only-export-components, LAUNCH_AUDIT.md P2-10).
export const ThemeModeContext = createContext<ThemeModeContextValue | null>(null);
