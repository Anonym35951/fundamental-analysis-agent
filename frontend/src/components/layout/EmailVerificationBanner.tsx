import { useEffect, useState } from "react";
import { MailWarning } from "lucide-react";
import { getCurrentUser, resendVerificationEmail } from "../../api/auth";
import { theme } from "../ui/theme";

/** Hinweis-Banner im App-Bereich, solange die E-Mail-Adresse des Kontos noch
 * nicht bestätigt ist. Analysen sind serverseitig gesperrt, bis die
 * Bestätigung erfolgt — der Banner erklärt das und bietet den erneuten
 * Versand des Links an. */
function EmailVerificationBanner() {
  const [isUnverified, setIsUnverified] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    getCurrentUser()
      .then((user) => {
        if (isMounted) setIsUnverified(user.email_verified === false);
      })
      .catch(() => {
        // Kein Banner, wenn der User-Status nicht ladbar ist — andere
        // Stellen (ProtectedRoute/API-Fehler) kümmern sich um Auth-Probleme.
      });

    return () => {
      isMounted = false;
    };
  }, []);

  if (!isUnverified) return null;

  async function handleResend() {
    setIsSending(true);
    setFeedback(null);
    try {
      await resendVerificationEmail();
      setFeedback("Bestätigungs-Mail wurde erneut gesendet. Bitte prüfe dein Postfach.");
    } catch {
      setFeedback(
        "Senden aktuell nicht möglich. Bitte warte einen Moment und versuche es erneut."
      );
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: "12px",
        padding: "14px 16px",
        marginBottom: "24px",
        borderRadius: theme.radius.md,
        background: theme.colors.chromeSoft,
        border: `1px solid ${theme.colors.chromeBorder}`,
        color: theme.colors.textSecondary,
        fontSize: "0.95rem",
        lineHeight: 1.6,
      }}
    >
      <MailWarning size={18} style={{ flexShrink: 0 }} />
      <span style={{ flex: 1, minWidth: "220px" }}>
        {feedback ??
          "Bitte bestätige deine E-Mail-Adresse über den Link, den wir dir geschickt haben — erst danach kannst du Analysen starten."}
      </span>
      <button
        type="button"
        onClick={handleResend}
        disabled={isSending}
        style={{
          padding: "8px 14px",
          borderRadius: theme.radius.md,
          border: `1px solid ${theme.colors.chromeBorder}`,
          background: "transparent",
          color: theme.colors.chrome,
          fontWeight: 700,
          fontSize: "0.9rem",
          cursor: isSending ? "wait" : "pointer",
        }}
      >
        {isSending ? "Wird gesendet…" : "Link erneut senden"}
      </button>
    </div>
  );
}

export default EmailVerificationBanner;
