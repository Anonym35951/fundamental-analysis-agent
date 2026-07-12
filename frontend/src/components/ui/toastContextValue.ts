import { createContext } from "react";

export type ToastTone = "success" | "error" | "info";

export type ToastContextValue = {
  showToast: (message: string, tone?: ToastTone) => void;
};

// Eigene Datei ohne Komponenten-Export, damit weder Toast.tsx (Provider)
// noch useToast.ts (Hook) selbst einen Nicht-Komponenten-Wert exportieren
// (react-refresh/only-export-components, LAUNCH_AUDIT.md P2-10).
export const ToastContext = createContext<ToastContextValue | null>(null);
