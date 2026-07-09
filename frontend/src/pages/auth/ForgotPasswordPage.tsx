import { useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle, CheckCircle2 } from "lucide-react";
import { requestPasswordReset } from "../../api/auth";
import { theme } from "../../components/ui/theme";
import Input from "../../components/ui/Input";
import Button from "../../components/ui/Button";

function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setErrorMessage("");

    if (!email.trim()) {
      setErrorMessage("Bitte gib deine E-Mail ein.");
      return;
    }

    setIsSubmitting(true);

    try {
      await requestPasswordReset(email.trim());
      setSubmitted(true);
    } catch (error) {
      if (error instanceof Error) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Anfrage fehlgeschlagen.");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div>
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
          Passwort vergessen?
        </h2>

        <p
          style={{
            margin: 0,
            color: theme.colors.textSecondary,
            fontSize: "1.05rem",
            lineHeight: 1.8,
          }}
        >
          Gib deine E-Mail-Adresse ein, wir senden dir einen Link zum Zurücksetzen.
        </p>
      </div>

      {submitted ? (
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "10px",
            padding: "14px 16px",
            borderRadius: theme.radius.md,
            background: theme.colors.successSoft,
            border: `1px solid ${theme.colors.successBorder}`,
            color: theme.colors.successText,
            fontSize: "0.98rem",
            lineHeight: 1.7,
          }}
        >
          <CheckCircle2 size={18} style={{ flexShrink: 0, marginTop: "2px" }} />
          <span>
            Falls ein Konto mit dieser E-Mail existiert, wurde ein Link zum
            Zurücksetzen des Passworts versendet. Bitte prüfe dein Postfach.
          </span>
        </div>
      ) : (
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
              E-Mail
            </label>

            <Input
              id="email"
              type="email"
              placeholder="deine@email.de"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              disabled={isSubmitting}
            />
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
                    alignItems: "flex-start",
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
                  <AlertCircle size={18} style={{ flexShrink: 0, marginTop: "2px" }} />
                  <span>{errorMessage}</span>
                </div>
              </motion.div>
            ) : null}
          </AnimatePresence>

          <Button
            type="submit"
            variant="cta"
            disabled={isSubmitting}
            style={{ marginTop: "4px", width: "100%", padding: "16px 18px" }}
          >
            {isSubmitting ? "Wird gesendet..." : "Link senden"}
          </Button>
        </form>
      )}

      <div
        style={{
          marginTop: "22px",
          textAlign: "center",
          color: theme.colors.textSecondary,
          fontSize: "0.98rem",
          lineHeight: 1.7,
        }}
      >
        <Link
          to="/login"
          style={{
            color: theme.colors.chrome,
            fontWeight: 700,
            textDecoration: "none",
          }}
        >
          Zurück zum Login
        </Link>
      </div>
    </div>
  );
}

export default ForgotPasswordPage;
