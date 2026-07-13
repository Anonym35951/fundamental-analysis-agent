import { createContext } from "react";

export type ToastTone = "success" | "error" | "info";

export type ToastAction = { label: string; onClick: () => void };

export type ToastOptions = { durationMs?: number; action?: ToastAction };

export type ToastContextValue = {
  showToast: (message: string, tone?: ToastTone, options?: ToastOptions) => void;
};

// Eigene Datei ohne Komponenten-Export, damit weder Toast.tsx (Provider)
// noch useToast.ts (Hook) selbst einen Nicht-Komponenten-Wert exportieren
// (react-refresh/only-export-components, LAUNCH_AUDIT.md P2-10).
export const ToastContext = createContext<ToastContextValue | null>(null);
