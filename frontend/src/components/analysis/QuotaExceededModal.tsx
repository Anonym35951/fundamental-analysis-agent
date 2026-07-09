import { useNavigate } from "react-router-dom";
import { CalendarClock } from "lucide-react";
import Modal from "../ui/Modal";
import { theme } from "../ui/theme";

type QuotaExceededModalProps = {
  isOpen: boolean;
  onClose: () => void;
  /** ISO date (YYYY-MM-DD) of the next monthly reset, from the backend's
   * QUOTA_EXCEEDED detail — falls back to a generic wording if absent (e.g.
   * the pre-emptive client-side check, which doesn't have a server round-trip). */
  resetDate?: string | null;
};

function formatResetDate(resetDate?: string | null): string {
  if (!resetDate) return "zum Monatsanfang";
  const parsed = new Date(`${resetDate}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return "zum Monatsanfang";
  return parsed.toLocaleDateString("de-DE", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

/** Shown instead of a raw error/disabled-button when a Free-plan user hits
 * the monthly analysis quota — the one moment they actually need Pro, so it
 * doubles as an honest, pressure-free upgrade invitation rather than a dead
 * end. Reused both pre-emptively (client-side limit check before the request
 * goes out) and reactively (backend's QUOTA_EXCEEDED, e.g. race conditions
 * with parallel requests). */
export default function QuotaExceededModal({ isOpen, onClose, resetDate }: QuotaExceededModalProps) {
  const navigate = useNavigate();

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Monatliches Kontingent aufgebraucht">
      <p
        style={{
          margin: "0 0 14px 0",
          color: theme.colors.textSecondary,
          fontSize: "0.98rem",
          lineHeight: 1.7,
        }}
      >
        Du hast alle Analysen deines Free-Plans für diesen Monat genutzt. Am{" "}
        <strong style={{ color: theme.colors.textPrimary }}>{formatResetDate(resetDate)}</strong>{" "}
        stehen dir wieder neue Analysen zur Verfügung — oder du wechselst jetzt zu
        Pro und legst sofort weiter los.
      </p>

      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: "10px",
          padding: "12px 14px",
          marginBottom: "18px",
          borderRadius: theme.radius.md,
          background: theme.colors.chromeSoft,
          border: `1px solid ${theme.colors.chromeBorder}`,
          color: theme.colors.textSecondary,
          fontSize: "0.92rem",
          lineHeight: 1.6,
        }}
      >
        <CalendarClock size={18} style={{ flexShrink: 0, marginTop: "2px" }} />
        <span>Mit Pro sind Analysen unbegrenzt — keine Wartezeit bis zum nächsten Monat.</span>
      </div>

      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
        <button
          type="button"
          onClick={onClose}
          style={{
            flex: "1 1 auto",
            padding: "13px 18px",
            borderRadius: theme.radius.md,
            border: `1px solid ${theme.glass.subtle.border}`,
            background: "transparent",
            color: theme.colors.textSecondary,
            fontWeight: 700,
            fontSize: "0.95rem",
            cursor: "pointer",
          }}
        >
          Später
        </button>
        <button
          type="button"
          onClick={() => navigate("/app/billing")}
          style={{
            flex: "1 1 auto",
            padding: "13px 18px",
            borderRadius: theme.radius.md,
            border: "none",
            background: theme.colors.chrome,
            color: theme.colors.black,
            fontWeight: 800,
            fontSize: "0.95rem",
            cursor: "pointer",
          }}
        >
          Zu Pro wechseln
        </button>
      </div>
    </Modal>
  );
}
