import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { verifyEmail } from "../../api/auth";
import { theme } from "../../components/ui/theme";

type VerifyState = "verifying" | "success" | "error";

function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");

  const [state, setState] = useState<VerifyState>(token ? "verifying" : "error");
  // React 18 StrictMode mountet Effekte doppelt — der zweite Aufruf würde mit
  // dem bereits verbrauchten (single-use) Token fehlschlagen und dem Nutzer
  // fälschlich einen Fehler zeigen.
  const hasRequested = useRef(false);

  useEffect(() => {
    if (!token || hasRequested.current) return;
    hasRequested.current = true;

    verifyEmail(token)
      .then(() => setState("success"))
      .catch(() => setState("error"));
  }, [token]);

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
          E-Mail bestätigen
        </h2>
      </div>

      {state === "verifying" ? (
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "10px",
            padding: "14px 16px",
            borderRadius: theme.radius.md,
            background: theme.colors.chromeSoft,
            border: `1px solid ${theme.colors.chromeBorder}`,
            color: theme.colors.textSecondary,
            fontSize: "0.98rem",
            lineHeight: 1.7,
          }}
        >
          <Loader2
            size={18}
            style={{ flexShrink: 0, marginTop: "2px", animation: "spin 1s linear infinite" }}
          />
          <span>Deine E-Mail-Adresse wird bestätigt…</span>
        </div>
      ) : null}

      {state === "success" ? (
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
            Deine E-Mail-Adresse wurde bestätigt. Du kannst jetzt alle
            Analysen nutzen.
          </span>
        </div>
      ) : null}

      {state === "error" ? (
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
            Dieser Bestätigungslink ist ungültig oder abgelaufen. Melde dich
            an und fordere in deinem Konto einen neuen Link an.
          </span>
        </div>
      ) : null}

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
          Zum Login
        </Link>
      </div>
    </div>
  );
}

export default VerifyEmailPage;
