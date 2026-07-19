import {
  useCallback,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { X } from "lucide-react";
import { theme } from "./theme";
import { useIsMobile } from "../../hooks/useMediaQuery";
import { ToastContext, type ToastAction, type ToastTone } from "./toastContextValue";
import { useTranslation } from "../../i18n/useTranslation";

type ToastItem = {
  id: number;
  message: string;
  tone: ToastTone;
  durationMs: number;
  action?: ToastAction;
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
  const { t } = useTranslation("common");

  const dismissToast = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const showToast = useCallback(
    (message: string, tone: ToastTone = "info", options?: { durationMs?: number; action?: ToastAction }) => {
      const id = nextId.current++;
      const durationMs = options?.durationMs ?? AUTO_DISMISS_MS;
      setToasts((current) => [...current, { id, message, tone, durationMs, action: options?.action }]);
      window.setTimeout(() => dismissToast(id), durationMs);
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

          if (toast.action) {
            return (
              <div
                key={toast.id}
                style={{
                  position: "relative",
                  overflow: "hidden",
                  padding: "18px 18px 14px",
                  borderRadius: theme.radius.sm,
                  background: style.background,
                  border: `1px solid ${style.border}`,
                  color: style.color,
                  fontSize: "0.92rem",
                  maxWidth: isMobile ? "none" : "400px",
                  boxShadow: "0 10px 30px rgba(0, 0, 0, 0.5)",
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    height: "3px",
                    width: "100%",
                    transformOrigin: "left",
                    background: theme.colors.chrome,
                    animation: `toast-progress-fill ${toast.durationMs}ms linear forwards`,
                  }}
                />
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    dismissToast(toast.id);
                  }}
                  aria-label={t("toast.dismissAriaLabel")}
                  style={{
                    position: "absolute",
                    top: "10px",
                    right: "10px",
                    background: "none",
                    border: "none",
                    color: style.color,
                    opacity: 0.7,
                    cursor: "pointer",
                    padding: "4px",
                    display: "flex",
                  }}
                >
                  <X size={16} />
                </button>
                <div style={{ paddingRight: "26px", lineHeight: 1.5 }}>{toast.message}</div>
                <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "12px" }}>
                  <button
                    type="button"
                    onClick={() => {
                      toast.action?.onClick();
                      dismissToast(toast.id);
                    }}
                    style={{
                      padding: "8px 14px",
                      borderRadius: theme.radius.sm,
                      background: "transparent",
                      border: `1px solid ${style.border}`,
                      color: style.color,
                      fontSize: "0.86rem",
                      fontWeight: 700,
                      cursor: "pointer",
                    }}
                  >
                    {toast.action.label}
                  </button>
                </div>
              </div>
            );
          }

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
