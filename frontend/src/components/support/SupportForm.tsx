import { useEffect, useState, type CSSProperties } from "react";
import { CheckCircle2 } from "lucide-react";
import { ApiError } from "../../api/client";
import { sendSupportRequest, type SupportCategory } from "../../api/support";
import { theme } from "../ui/theme";
import Input from "../ui/Input";
import Select from "../ui/Select";
import Button from "../ui/Button";
import { useToast } from "../ui/Toast";

// Muss mit api/schemas/support.py (SUPPORT_CATEGORIES) synchron bleiben.
const SUPPORT_CATEGORIES: SupportCategory[] = [
  "Allgemeine Frage",
  "Technisches Problem",
  "Abrechnung & Abo",
  "Feedback",
  "Sonstiges",
];

type SupportFormProps = {
  /** Vorausgefuellte E-Mail fuer eingeloggte Nutzer (Feld bleibt editierbar). */
  prefilledEmail?: string;
};

function SupportForm({ prefilledEmail }: SupportFormProps) {
  const { showToast } = useToast();

  const [category, setCategory] = useState<SupportCategory>(SUPPORT_CATEGORIES[0]);
  const [email, setEmail] = useState(prefilledEmail ?? "");
  const [message, setMessage] = useState("");

  // prefilledEmail kommt bei eingeloggten Nutzern erst asynchron nach dem
  // ersten Render an (SupportPage laedt sie via getCurrentUser()) - der
  // useState-Initializer oben greift nur beim allerersten Mount, daher hier
  // zusaetzlich synchronisieren, sobald der Wert eintrifft. Nur setzen,
  // solange das Feld noch leer ist, damit ein bereits eingetipptes Feld
  // nicht überschrieben wird.
  useEffect(() => {
    if (prefilledEmail && !email) {
      setEmail(prefilledEmail);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefilledEmail]);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitted, setIsSubmitted] = useState(false);

  const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setErrorMessage("");

    if (!emailPattern.test(email.trim())) {
      setErrorMessage("Bitte gib eine gültige E-Mail-Adresse ein.");
      return;
    }
    if (message.trim().length < 10) {
      setErrorMessage("Deine Nachricht sollte mindestens 10 Zeichen umfassen.");
      return;
    }

    setIsSubmitting(true);
    try {
      await sendSupportRequest({ category, email: email.trim(), message: message.trim() });
      setIsSubmitted(true);
      showToast("Deine Anfrage wurde gesendet.", "success");
    } catch (error) {
      if (error instanceof ApiError && error.status === 429) {
        setErrorMessage("Zu viele Anfragen. Bitte versuche es in einer Minute erneut.");
      } else if (error instanceof Error) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Anfrage konnte nicht gesendet werden.");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  if (isSubmitted) {
    return (
      <div style={successCard}>
        <CheckCircle2 size={22} color={theme.colors.successText} style={{ flexShrink: 0 }} />
        <div>
          <div style={successTitle}>Vielen Dank!</div>
          <p style={successText}>
            Deine Anfrage wurde gesendet. Wir melden uns in der Regel innerhalb von
            1–2 Werktagen.
          </p>
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} style={{ display: "grid", gap: "18px" }}>
      <div>
        <label style={fieldLabel} htmlFor="support-category">
          Kategorie
        </label>
        <Select
          id="support-category"
          value={category}
          onChange={(event) => setCategory(event.target.value as SupportCategory)}
          disabled={isSubmitting}
        >
          {SUPPORT_CATEGORIES.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </Select>
      </div>

      <div>
        <label style={fieldLabel} htmlFor="support-email">
          E-Mail-Adresse
        </label>
        <Input
          id="support-email"
          type="email"
          placeholder="deine@email.de"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          disabled={isSubmitting}
        />
      </div>

      <div>
        <label style={fieldLabel} htmlFor="support-message">
          Nachricht
        </label>
        <textarea
          id="support-message"
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          disabled={isSubmitting}
          rows={6}
          placeholder="Beschreibe dein Anliegen..."
          style={textareaStyle}
        />
      </div>

      {errorMessage ? <div style={errorBox}>{errorMessage}</div> : null}

      <Button type="submit" variant="cta" disabled={isSubmitting} style={{ justifySelf: "start" }}>
        {isSubmitting ? "Wird gesendet..." : "Anfrage senden"}
      </Button>
    </form>
  );
}

const fieldLabel: CSSProperties = {
  display: "block",
  marginBottom: "8px",
  fontWeight: 700,
  color: theme.colors.textSecondary,
  fontSize: "0.85rem",
  textTransform: "uppercase",
  letterSpacing: "0.06em",
};

const textareaStyle: CSSProperties = {
  width: "100%",
  padding: "11px 16px",
  borderRadius: theme.radius.md,
  border: `1px solid ${theme.glass.subtle.border}`,
  background: theme.glass.subtle.background,
  color: theme.colors.textPrimary,
  fontSize: "0.95rem",
  fontFamily: "inherit",
  outline: "none",
  boxSizing: "border-box",
  resize: "vertical",
};

const errorBox: CSSProperties = {
  padding: "12px 14px",
  borderRadius: theme.radius.md,
  background: theme.colors.dangerSoft,
  border: `1px solid ${theme.colors.dangerBorder}`,
  color: theme.colors.dangerText,
  fontSize: "0.92rem",
  lineHeight: 1.6,
};

const successCard: CSSProperties = {
  display: "flex",
  alignItems: "flex-start",
  gap: "14px",
  padding: "20px 22px",
  borderRadius: theme.radius.lg,
  background: theme.colors.successSoft,
  border: `1px solid ${theme.colors.successBorder}`,
};

const successTitle: CSSProperties = {
  fontWeight: 700,
  color: theme.colors.successText,
  marginBottom: "4px",
};

const successText: CSSProperties = {
  margin: 0,
  color: theme.colors.textSecondary,
  fontSize: "0.95rem",
  lineHeight: 1.7,
};

export default SupportForm;
