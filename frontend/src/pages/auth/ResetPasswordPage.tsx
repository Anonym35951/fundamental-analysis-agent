import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle } from "lucide-react";
import { resetPassword } from "../../api/auth";
import { theme } from "../../components/ui/theme";
import Input from "../../components/ui/Input";
import Button from "../../components/ui/Button";

function ResetPasswordPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");

  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setErrorMessage("");

    if (!token) {
      setErrorMessage("Der Link ist ungültig. Bitte fordere einen neuen Reset-Link an.");
      return;
    }

    if (!newPassword || newPassword.length < 8) {
      setErrorMessage("Das Passwort muss mindestens 8 Zeichen lang sein.");
      return;
    }

    if (newPassword !== confirmPassword) {
      setErrorMessage("Die Passwörter stimmen nicht überein.");
      return;
    }

    setIsSubmitting(true);

    try {
      await resetPassword(token, newPassword);
      sessionStorage.setItem("password_reset_success", "1");
      navigate("/login");
    } catch (error) {
      if (error instanceof Error) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Passwort konnte nicht zurückgesetzt werden.");
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
          Neues Passwort vergeben
        </h2>

        <p
          style={{
            margin: 0,
            color: theme.colors.textSecondary,
            fontSize: "1.05rem",
            lineHeight: 1.8,
          }}
        >
          Wähle ein neues Passwort für dein Konto.
        </p>
      </div>

      {!token ? (
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "10px",
            padding: "14px 16px",
            borderRadius: theme.radius.md,
            background: theme.colors.dangerSoft,
            border: `1px solid ${theme.colors.dangerBorder}`,
            color: theme.colors.dangerText,
            fontSize: "0.98rem",
            lineHeight: 1.7,
          }}
        >
          <AlertCircle size={18} style={{ flexShrink: 0, marginTop: "2px" }} />
          <span>
            Dieser Link ist ungültig oder unvollständig. Bitte fordere einen
            neuen Reset-Link an.
          </span>
        </div>
      ) : (
        <form onSubmit={handleSubmit} style={{ display: "grid", gap: "18px" }}>
          <div>
            <label
              htmlFor="newPassword"
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
              Neues Passwort
            </label>

            <Input
              id="newPassword"
              type="password"
              placeholder="••••••••"
              value={newPassword}
              onChange={(event) => setNewPassword(event.target.value)}
              disabled={isSubmitting}
            />
          </div>

          <div>
            <label
              htmlFor="confirmPassword"
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
              Passwort bestätigen
            </label>

            <Input
              id="confirmPassword"
              type="password"
              placeholder="••••••••"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
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
            {isSubmitting ? "Wird gespeichert..." : "Passwort zurücksetzen"}
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

export default ResetPasswordPage;
