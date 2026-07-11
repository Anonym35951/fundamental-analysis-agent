import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { theme } from "./theme";

type ToastTone = "success" | "error" | "info";

type ToastItem = {
  id: number;
  message: string;
  tone: ToastTone;
};

type ToastContextValue = {
  showToast: (message: string, tone?: ToastTone) => void;
};

const ToastContext = createContext<ToastContextValue | null>(null);

const toneStyles: Record<ToastTone, { background: string; border: string; color: string }> = {
  success: {
    background: "rgba(10, 10, 11, 0.95)",
    border: theme.colors.successBorder,
    color: "#a7f3d0",
  },
  error: {
    background: "rgba(10, 10, 11, 0.95)",
    border: theme.colors.dangerBorder,
    color: "#fecaca",
  },
  info: {
    background: "rgba(10, 10, 11, 0.95)",
    border: theme.colors.chromeBorder,
    color: "#f5f5f7",
  },
};

const AUTO_DISMISS_MS = 4000;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const nextId = useRef(0);

  const dismissToast = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const showToast = useCallback(
    (message: string, tone: ToastTone = "info") => {
      const id = nextId.current++;
      setToasts((current) => [...current, { id, message, tone }]);
      window.setTimeout(() => dismissToast(id), AUTO_DISMISS_MS);
    },
    [dismissToast]
  );

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}

      <div
        style={{
          position: "fixed",
          // Zusätzlicher Puffer für den iOS-Home-Indicator (0 auf Geräten
          // ohne Safe-Area-Inset) — siehe RESPONSIVE.md R-P0-5.
          bottom: "calc(24px + env(safe-area-inset-bottom))",
          right: "24px",
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          zIndex: 1000,
        }}
      >
        {toasts.map((toast) => {
          const style = toneStyles[toast.tone];
          return (
            <div
              key={toast.id}
              onClick={() => dismissToast(toast.id)}
              style={{
                padding: "14px 18px",
                borderRadius: theme.radius.sm,
                background: style.background,
                border: `1px solid ${style.border}`,
                color: style.color,
                fontSize: "0.92rem",
                maxWidth: "360px",
                boxShadow: "0 10px 30px rgba(0, 0, 0, 0.5)",
                cursor: "pointer",
              }}
            >
              {toast.message}
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
}
