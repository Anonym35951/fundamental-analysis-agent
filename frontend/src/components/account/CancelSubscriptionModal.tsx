import { useEffect, useState } from "react";
import { getUsageSummary, type UsageSummary } from "../../api/account";
import { useIsMobile } from "../../hooks/useMediaQuery";
import Modal from "../ui/Modal";
import { theme } from "../ui/theme";

type CancelSubscriptionModalProps = {
  isOpen: boolean;
  isLoading: boolean;
  onConfirm: (reason: string) => void;
  onCancel: () => void;
};

/** Kündigungs-Dialog mit neutralem Nutzungs-Rückblick + optionaler,
 * überspringbarer Grund-Frage (siehe [[comanalysis-product-decisions]] /
 * Welle 4). Reiner Rückblick, kein Guilt-Tripping — der Rückblick wird nur
 * gezeigt, wenn der Nutzer überhaupt schon Analysen ausgeführt hat. */
function CancelSubscriptionModal({
  isOpen,
  isLoading,
  onConfirm,
  onCancel,
}: CancelSubscriptionModalProps) {
  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [reason, setReason] = useState("");
  const isMobile = useIsMobile();
  // Auf Mobile gestapelt statt nebeneinander — siehe RESPONSIVE.md R-P0-2.
  const buttonRowStyle = isMobile ? buttonRowMobile : buttonRow;
  const fullWidthOnMobile = isMobile ? { width: "100%" } : undefined;

  useEffect(() => {
    if (!isOpen) return;

    setSummary(null);
    setReason("");

    let isCancelled = false;
    getUsageSummary()
      .then((result) => {
        if (!isCancelled) setSummary(result);
      })
      .catch(() => {
        // Rückblick ist ein Nice-to-have — bei Fehler einfach ohne ihn fortfahren.
      });

    return () => {
      isCancelled = true;
    };
  }, [isOpen]);

  return (
    <Modal isOpen={isOpen} onClose={onCancel} title="Abo wirklich kündigen?" maxWidth="480px">
      <p style={messageStyle}>
        Möchtest du dein Pro-Abonnement wirklich kündigen? Dein Zugang bleibt
        bis zum Ende des aktuellen Abrechnungszeitraums bestehen und endet
        danach automatisch.
      </p>

      {summary && summary.total_analyses > 0 ? (
        <div style={summaryBox}>
          Du hast bisher <strong>{summary.total_analyses}</strong>{" "}
          {summary.total_analyses === 1 ? "Analyse" : "Analysen"} über{" "}
          <strong>{summary.distinct_symbols}</strong>{" "}
          {summary.distinct_symbols === 1 ? "Unternehmen" : "Unternehmen"} ausgeführt.
        </div>
      ) : null}

      <div style={fieldGroup}>
        <label htmlFor="cancel-reason" style={fieldLabel}>
          Was ist der Grund? (optional)
        </label>
        <textarea
          id="cancel-reason"
          value={reason}
          onChange={(event) => setReason(event.target.value)}
          placeholder="z. B. zu teuer, zu wenig genutzt, fehlende Funktion... — hilft uns, ComAnalysis zu verbessern."
          rows={3}
          maxLength={500}
          style={textareaStyle}
        />
      </div>

      <div style={buttonRowStyle}>
        <button
          onClick={onCancel}
          disabled={isLoading}
          style={{
            ...cancelButton,
            ...fullWidthOnMobile,
            cursor: isLoading ? "not-allowed" : "pointer",
            opacity: isLoading ? 0.7 : 1,
          }}
        >
          Doch nicht
        </button>

        <button
          onClick={() => onConfirm(reason.trim())}
          disabled={isLoading}
          style={{
            ...confirmButton,
            ...fullWidthOnMobile,
            cursor: isLoading ? "not-allowed" : "pointer",
            opacity: isLoading ? 0.7 : 1,
          }}
        >
          {isLoading ? "Wird gekündigt..." : "Abo kündigen"}
        </button>
      </div>
    </Modal>
  );
}

export default CancelSubscriptionModal;

/* Styles */

const messageStyle: React.CSSProperties = {
  margin: "0 0 16px 0",
  fontSize: "0.98rem",
  lineHeight: 1.7,
  color: theme.colors.textSecondary,
};

const summaryBox: React.CSSProperties = {
  marginBottom: "18px",
  padding: "12px 14px",
  borderRadius: theme.radius.md,
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textPrimary,
  fontSize: "0.94rem",
  lineHeight: 1.6,
};

const fieldGroup: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "8px",
  marginBottom: "20px",
};

const fieldLabel: React.CSSProperties = {
  color: theme.colors.textSecondary,
  fontSize: "0.9rem",
  fontWeight: 700,
};

const textareaStyle: React.CSSProperties = {
  width: "100%",
  padding: "12px 14px",
  borderRadius: theme.radius.md,
  border: "1px solid rgba(148, 163, 184, 0.18)",
  background: theme.colors.panelAlt,
  color: theme.colors.textPrimary,
  // >= 16px, sonst zoomt iOS Safari beim Fokussieren die Seite.
  fontSize: "max(16px, 0.94rem)",
  fontFamily: "inherit",
  outline: "none",
  resize: "vertical",
  boxSizing: "border-box",
};

const buttonRow: React.CSSProperties = {
  display: "flex",
  justifyContent: "flex-end",
  gap: "10px",
};

// Auf Mobile gestapelt statt nebeneinander: volle Touch-Breite, die primäre
// Aktion (letztes DOM-Kind) landet dadurch unten, näher am Daumen.
const buttonRowMobile: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "10px",
};

const cancelButton: React.CSSProperties = {
  padding: "10px 14px",
  borderRadius: "999px",
  background: "transparent",
  color: theme.colors.textSecondary,
  border: `1px solid ${theme.glass.subtle.border}`,
  fontWeight: 700,
};

const confirmButton: React.CSSProperties = {
  padding: "10px 14px",
  borderRadius: "999px",
  background: "linear-gradient(135deg, #b91c1c, #dc2626)",
  color: "#ffffff",
  border: "none",
  fontWeight: 800,
  boxShadow: "0 10px 24px rgba(220, 38, 38, 0.25)",
};
