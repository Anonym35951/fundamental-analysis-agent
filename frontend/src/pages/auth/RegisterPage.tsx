import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle } from "lucide-react";
import { registerUser } from "../../api/auth";
import { useToast } from "../../components/ui/Toast";
import { theme } from "../../components/ui/theme";
import Input from "../../components/ui/Input";
import Button from "../../components/ui/Button";

const MIN_AGE = 16;
const USERNAME_PATTERN = /^[a-zA-Z0-9_.-]{3,50}$/;

function RegisterPage() {
  const navigate = useNavigate();
  const { showToast } = useToast();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [username, setUsername] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [age, setAge] = useState("");
  const [termsAccepted, setTermsAccepted] = useState(false);
  const [privacyAccepted, setPrivacyAccepted] = useState(false);

  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setErrorMessage("");

    if (!email.trim()) {
      setErrorMessage("Bitte gib deine E-Mail ein.");
      return;
    }

    if (!password) {
      setErrorMessage("Bitte gib ein Passwort ein.");
      return;
    }

    if (password.length < 8) {
      setErrorMessage("Das Passwort muss mindestens 8 Zeichen lang sein.");
      return;
    }

    if (password !== confirmPassword) {
      setErrorMessage("Die Passwörter stimmen nicht überein.");
      return;
    }

    if (!USERNAME_PATTERN.test(username.trim())) {
      setErrorMessage(
        "Der Benutzername muss 3-50 Zeichen lang sein und darf nur Buchstaben, Zahlen, Punkt, Unterstrich oder Bindestrich enthalten."
      );
      return;
    }

    if (!firstName.trim() || !lastName.trim()) {
      setErrorMessage("Bitte gib deinen Vor- und Nachnamen ein.");
      return;
    }

    const ageNumber = Number(age);
    if (!age || Number.isNaN(ageNumber) || ageNumber < MIN_AGE) {
      setErrorMessage(`Du musst mindestens ${MIN_AGE} Jahre alt sein.`);
      return;
    }

    if (!termsAccepted) {
      setErrorMessage("Bitte akzeptiere die Nutzungsbedingungen.");
      return;
    }

    if (!privacyAccepted) {
      setErrorMessage(
        "Bitte bestätige, dass du die Datenschutzerklärung zur Kenntnis genommen hast."
      );
      return;
    }

    setIsSubmitting(true);

    try {
      await registerUser({
        email: email.trim(),
        password,
        username: username.trim(),
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        age: ageNumber,
        terms_accepted: termsAccepted,
        privacy_accepted: privacyAccepted,
      });

      showToast(
        "Registrierung erfolgreich. Wir haben dir einen Bestätigungslink per E-Mail geschickt.",
        "success"
      );
      setEmail("");
      setPassword("");
      setConfirmPassword("");
      setUsername("");
      setFirstName("");
      setLastName("");
      setAge("");
      setTermsAccepted(false);
      setPrivacyAccepted(false);

      sessionStorage.setItem("registration_verify_pending", "1");

      setTimeout(() => {
        navigate("/login");
      }, 1800);
    } catch (error) {
      if (error instanceof Error) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Registrierung fehlgeschlagen.");
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
          Konto erstellen
        </h2>

        <p
          style={{
            margin: 0,
            color: theme.colors.textSecondary,
            fontSize: "1.05rem",
            lineHeight: 1.8,
          }}
        >
          Registriere dich, um Analysen zu starten und dein persönliches
          Dashboard zu nutzen.
        </p>
      </div>

      <form onSubmit={handleSubmit} style={{ display: "grid", gap: "18px" }}>
        <div>
          <label htmlFor="email" style={labelStyle}>
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

        <div>
          <label htmlFor="username" style={labelStyle}>
            Benutzername
          </label>

          <Input
            id="username"
            type="text"
            placeholder="z. B. max_mustermann"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            disabled={isSubmitting}
          />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "18px" }}>
          <div>
            <label htmlFor="firstName" style={labelStyle}>
              Vorname
            </label>

            <Input
              id="firstName"
              type="text"
              placeholder="Max"
              value={firstName}
              onChange={(event) => setFirstName(event.target.value)}
              disabled={isSubmitting}
            />
          </div>

          <div>
            <label htmlFor="lastName" style={labelStyle}>
              Nachname
            </label>

            <Input
              id="lastName"
              type="text"
              placeholder="Mustermann"
              value={lastName}
              onChange={(event) => setLastName(event.target.value)}
              disabled={isSubmitting}
            />
          </div>
        </div>

        <div>
          <label htmlFor="age" style={labelStyle}>
            Alter
          </label>

          <Input
            id="age"
            type="number"
            min={MIN_AGE}
            placeholder="Mindestens 16"
            value={age}
            onChange={(event) => setAge(event.target.value)}
            disabled={isSubmitting}
          />
        </div>

        <div>
          <label htmlFor="password" style={labelStyle}>
            Passwort
          </label>

          <Input
            id="password"
            type="password"
            placeholder="Mindestens 8 Zeichen"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            disabled={isSubmitting}
          />
        </div>

        <div>
          <label htmlFor="confirmPassword" style={labelStyle}>
            Passwort bestätigen
          </label>

          <Input
            id="confirmPassword"
            type="password"
            placeholder="Passwort erneut eingeben"
            value={confirmPassword}
            onChange={(event) => setConfirmPassword(event.target.value)}
            disabled={isSubmitting}
          />
        </div>

        <label style={consentCheckboxStyle}>
          <input
            type="checkbox"
            checked={termsAccepted}
            onChange={(event) => setTermsAccepted(event.target.checked)}
            disabled={isSubmitting}
            style={{ marginTop: "3px", flexShrink: 0 }}
          />
          <span>
            Ich akzeptiere die{" "}
            <Link
              to="/legal/terms"
              target="_blank"
              style={{ color: theme.colors.chrome, fontWeight: 700 }}
            >
              Nutzungsbedingungen
            </Link>
            .
          </span>
        </label>

        <label style={consentCheckboxStyle}>
          <input
            type="checkbox"
            checked={privacyAccepted}
            onChange={(event) => setPrivacyAccepted(event.target.checked)}
            disabled={isSubmitting}
            style={{ marginTop: "3px", flexShrink: 0 }}
          />
          <span>
            Ich habe die{" "}
            <Link
              to="/legal/privacy"
              target="_blank"
              style={{ color: theme.colors.chrome, fontWeight: 700 }}
            >
              Datenschutzerklärung
            </Link>{" "}
            zur Kenntnis genommen und stimme der Verarbeitung meiner Daten gemäß
            dieser Erklärung zu.
          </span>
        </label>

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

        <Button type="submit" variant="cta" disabled={isSubmitting} style={{ marginTop: "4px", width: "100%", padding: "16px 18px" }}>
          {isSubmitting ? "Registrierung läuft..." : "Registrieren"}
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
        Bereits ein Konto?{" "}
        <Link
          to="/login"
          style={{
            color: theme.colors.chrome,
            fontWeight: 700,
            textDecoration: "none",
          }}
        >
          Jetzt einloggen
        </Link>
      </div>
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: "block",
  marginBottom: "8px",
  fontWeight: 700,
  color: theme.colors.textSecondary,
  fontSize: "0.85rem",
  textTransform: "uppercase",
  letterSpacing: "0.06em",
};

const consentCheckboxStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "flex-start",
  gap: "10px",
  padding: "12px 14px",
  borderRadius: theme.radius.md,
  border: `1px solid ${theme.glass.subtle.border}`,
  background: theme.glass.subtle.background,
  color: theme.colors.textSecondary,
  fontSize: "0.92rem",
  lineHeight: 1.6,
  cursor: "pointer",
};

export default RegisterPage;
