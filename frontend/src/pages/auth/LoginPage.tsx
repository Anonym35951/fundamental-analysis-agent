import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle, CheckCircle2, Clock, MailCheck } from "lucide-react";
import { ApiError } from "../../api/client";
import { loginUser, resendVerificationEmailPublic } from "../../api/auth";
import { theme } from "../../components/ui/theme";
import Input from "../../components/ui/Input";
import Button from "../../components/ui/Button";

function LoginPage() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Login schlägt für unverifizierte Konten serverseitig fehl (siehe
  // /auth/login) - zeigt zusätzlich zur Fehlermeldung eine Möglichkeit, den
  // Bestätigungslink erneut anzufordern, ohne dass dafür ein gültiger Token
  // nötig wäre (der ja gerade fehlt).
  const [isEmailNotVerified, setIsEmailNotVerified] = useState(false);
  const [isResendingVerification, setIsResendingVerification] = useState(false);
  const [resendFeedback, setResendFeedback] = useState("");

  // Sticky notice set by SessionTimeoutWatcher right before it logs the user
  // out for inactivity — read once on mount and cleared immediately so a
  // page refresh doesn't keep re-showing it; the banner itself only
  // disappears once the user clicks it, not automatically.
  const [showInactivityNotice, setShowInactivityNotice] = useState(() => {
    const wasLoggedOutForInactivity = sessionStorage.getItem("logged_out_reason") === "inactivity";
    if (wasLoggedOutForInactivity) sessionStorage.removeItem("logged_out_reason");
    return wasLoggedOutForInactivity;
  });

  // Sticky notice set by ResetPasswordPage right after a successful reset —
  // read once on mount and cleared immediately, same pattern as the
  // inactivity notice above.
  const [showPasswordResetNotice, setShowPasswordResetNotice] = useState(() => {
    const wasPasswordReset = sessionStorage.getItem("password_reset_success") === "1";
    if (wasPasswordReset) sessionStorage.removeItem("password_reset_success");
    return wasPasswordReset;
  });

  // Sticky notice set by RegisterPage after successful registration — the
  // account needs email verification before analyses can run.
  const [showVerifyPendingNotice, setShowVerifyPendingNotice] = useState(() => {
    const isVerifyPending = sessionStorage.getItem("registration_verify_pending") === "1";
    if (isVerifyPending) sessionStorage.removeItem("registration_verify_pending");
    return isVerifyPending;
  });

  // Sticky notice set by AccountPage right after a successful account
  // deletion — same pattern as the notices above.
  const [showAccountDeletedNotice, setShowAccountDeletedNotice] = useState(() => {
    const wasAccountDeleted = sessionStorage.getItem("account_deleted_success") === "1";
    if (wasAccountDeleted) sessionStorage.removeItem("account_deleted_success");
    return wasAccountDeleted;
  });

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setErrorMessage("");
    setIsEmailNotVerified(false);
    setResendFeedback("");

    if (!email.trim()) {
      setErrorMessage("Bitte gib deine E-Mail oder deinen Benutzernamen ein.");
      return;
    }

    if (!password) {
      setErrorMessage("Bitte gib dein Passwort ein.");
      return;
    }

    setIsSubmitting(true);

    try {
      const result = await loginUser({
        username: email.trim(),
        password,
      });

      localStorage.setItem("access_token", result.access_token);
      localStorage.setItem("token_type", result.token_type);

      sessionStorage.setItem("show_intro", "1");
      navigate("/app/dashboard");
    } catch (error) {
      if (error instanceof ApiError && error.code === "EMAIL_NOT_VERIFIED") {
        setIsEmailNotVerified(true);
        setErrorMessage(error.message);
      } else if (error instanceof Error) {
        if (error.message === "Invalid credentials") {
          setErrorMessage("E-Mail/Benutzername oder Passwort sind nicht korrekt.");
        } else {
          setErrorMessage(error.message);
        }
      } else {
        setErrorMessage("Login fehlgeschlagen.");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleResendVerification() {
    if (isResendingVerification || !email.trim()) return;
    setIsResendingVerification(true);
    setResendFeedback("");
    try {
      await resendVerificationEmailPublic(email.trim());
      setResendFeedback("Falls das Konto existiert, haben wir einen neuen Bestätigungslink gesendet.");
    } catch {
      setResendFeedback("Senden aktuell nicht möglich. Bitte versuche es in Kürze erneut.");
    } finally {
      setIsResendingVerification(false);
    }
  }

  return (
    <div>
      {showInactivityNotice ? (
        <div
          role="button"
          tabIndex={0}
          onClick={() => setShowInactivityNotice(false)}
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
            fontSize: "0.95rem",
            lineHeight: 1.6,
            cursor: "pointer",
          }}
        >
          <Clock size={18} style={{ flexShrink: 0, marginTop: "2px" }} />
          <span>Du wurdest wegen Inaktivität abgemeldet. Bitte melde dich erneut an. (Klicken, um zu schließen)</span>
        </div>
      ) : null}

      {showPasswordResetNotice ? (
        <div
          role="button"
          tabIndex={0}
          onClick={() => setShowPasswordResetNotice(false)}
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "10px",
            padding: "12px 14px",
            marginBottom: "18px",
            borderRadius: theme.radius.md,
            background: theme.colors.successSoft,
            border: `1px solid ${theme.colors.successBorder}`,
            color: theme.colors.successText,
            fontSize: "0.95rem",
            lineHeight: 1.6,
            cursor: "pointer",
          }}
        >
          <CheckCircle2 size={18} style={{ flexShrink: 0, marginTop: "2px" }} />
          <span>Dein Passwort wurde erfolgreich zurückgesetzt. Du kannst dich jetzt anmelden. (Klicken, um zu schließen)</span>
        </div>
      ) : null}

      {showVerifyPendingNotice ? (
        <div
          role="button"
          tabIndex={0}
          onClick={() => setShowVerifyPendingNotice(false)}
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
            fontSize: "0.95rem",
            lineHeight: 1.6,
            cursor: "pointer",
          }}
        >
          <MailCheck size={18} style={{ flexShrink: 0, marginTop: "2px" }} />
          <span>
            Wir haben dir einen Bestätigungslink per E-Mail geschickt. Bitte
            bestätige deine E-Mail-Adresse, um Analysen starten zu können.
            (Klicken, um zu schließen)
          </span>
        </div>
      ) : null}

      {showAccountDeletedNotice ? (
        <div
          role="button"
          tabIndex={0}
          onClick={() => setShowAccountDeletedNotice(false)}
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "10px",
            padding: "12px 14px",
            marginBottom: "18px",
            borderRadius: theme.radius.md,
            background: theme.colors.successSoft,
            border: `1px solid ${theme.colors.successBorder}`,
            color: theme.colors.successText,
            fontSize: "0.95rem",
            lineHeight: 1.6,
            cursor: "pointer",
          }}
        >
          <CheckCircle2 size={18} style={{ flexShrink: 0, marginTop: "2px" }} />
          <span>
            Dein Konto wurde endgültig gelöscht. Alle deine Daten wurden
            entfernt. (Klicken, um zu schließen)
          </span>
        </div>
      ) : null}

      <div style={{ marginBottom: "24px", textAlign: "center" }}>
        <h2
          style={{
            margin: "0 0 10px 0",
            fontSize: "2rem",
            lineHeight: 1.15,
            letterSpacing: "-0.03em",
            color: theme.colors.textPrimary,
          }}
        >
          Willkommen zurück
        </h2>

        <p
          style={{
            margin: 0,
            color: theme.colors.textSecondary,
            fontSize: "1.05rem",
            lineHeight: 1.8,
          }}
        >
          Melde dich an, um auf dein Analyse-Dashboard zuzugreifen.
        </p>
      </div>

      <form onSubmit={handleSubmit} style={{ display: "grid", gap: "18px" }}>
        <div>
          <label
            htmlFor="email"
            style={{
              display: "block",
              marginBottom: "8px",
              fontWeight: 700,
              color: theme.colors.textSecondary,
              fontSize: "0.85rem",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            E-Mail oder Benutzername
          </label>

          <Input
            id="email"
            type="text"
            placeholder="deine@email.de oder Benutzername"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            disabled={isSubmitting}
          />
        </div>

        <div>
          <label
            htmlFor="password"
            style={{
              display: "block",
              marginBottom: "8px",
              fontWeight: 700,
              color: theme.colors.textSecondary,
              fontSize: "0.85rem",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            Passwort
          </label>

          <Input
            id="password"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            disabled={isSubmitting}
          />

          <div style={{ marginTop: "8px", textAlign: "right" }}>
            <Link
              to="/forgot-password"
              style={{
                color: theme.colors.chrome,
                fontWeight: 700,
                fontSize: "0.88rem",
                textDecoration: "none",
              }}
            >
              Passwort vergessen?
            </Link>
          </div>
        </div>

        <AnimatePresence>
          {errorMessage ? (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              style={{ overflow: "hidden" }}
            >
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "10px",
                  padding: "12px 14px",
                  borderRadius: theme.radius.md,
                  background: theme.colors.dangerSoft,
                  border: `1px solid ${theme.colors.dangerBorder}`,
                  color: theme.colors.dangerText,
                  fontSize: "0.95rem",
                  lineHeight: 1.6,
                }}
              >
                <div style={{ display: "flex", alignItems: "flex-start", gap: "10px" }}>
                  <AlertCircle size={18} style={{ flexShrink: 0, marginTop: "2px" }} />
                  <span>{errorMessage}</span>
                </div>

                {isEmailNotVerified ? (
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                    <button
                      type="button"
                      onClick={handleResendVerification}
                      disabled={isResendingVerification}
                      style={{
                        padding: "8px 14px",
                        borderRadius: theme.radius.md,
                        border: `1px solid ${theme.colors.dangerBorder}`,
                        background: "transparent",
                        color: theme.colors.dangerText,
                        fontWeight: 700,
                        fontSize: "0.88rem",
                        cursor: isResendingVerification ? "wait" : "pointer",
                      }}
                    >
                      {isResendingVerification ? "Wird gesendet…" : "Bestätigungslink erneut senden"}
                    </button>
                    {resendFeedback ? (
                      <span style={{ fontSize: "0.88rem" }}>{resendFeedback}</span>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>

        <Button type="submit" variant="cta" disabled={isSubmitting} style={{ marginTop: "4px", width: "100%", padding: "16px 18px" }}>
          {isSubmitting ? "Login läuft..." : "Einloggen"}
        </Button>
      </form>

      <div
        style={{
          marginTop: "22px",
          textAlign: "center",
          color: theme.colors.textSecondary,
          fontSize: "0.98rem",
          lineHeight: 1.7,
        }}
      >
        Noch kein Konto?{" "}
        <Link
          to="/register"
          style={{
            color: theme.colors.chrome,
            fontWeight: 700,
            textDecoration: "none",
          }}
        >
          Jetzt registrieren
        </Link>
      </div>
    </div>
  );
}

export default LoginPage;
