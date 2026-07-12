import {
  useCallback,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { theme } from "./theme";
import { useIsMobile } from "../../hooks/useMediaQuery";
import { ToastContext, type ToastTone } from "./toastContextValue";

type ToastItem = {
  id: number;
  message: string;
  tone: ToastTone;
};

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
  const isMobile = useIsMobile();

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
          // Auf Mobile links+rechts statt rechtsbündig-fix: sonst klebt der
          // Toast auf schmalen Screens an der rechten Kante statt die volle
          // Breite auszunutzen (RESPONSIVE.md R-P1-10).
          ...(isMobile ? { left: "16px", right: "16px" } : { right: "24px" }),
          display: "flex",
          flexDirection: "column",
          alignItems: isMobile ? "stretch" : "flex-end",
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
                maxWidth: isMobile ? "none" : "360px",
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
