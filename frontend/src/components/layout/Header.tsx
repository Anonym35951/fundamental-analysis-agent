import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { theme } from "../ui/theme";
import { useLocale } from "../../i18n/useLocale";
import type { Locale } from "../../i18n/config";

type HeaderProps = {
  variant?: "public" | "auth";
};

// EVOLVING.md § Internationalisierung, I18N-006: kompakter DE/EN-Switcher,
// sichtbar auf allen öffentlichen und Auth-Seiten (unabhängig vom
// Login-Status) - schreibt nur Context + localStorage (i18n/LocaleProvider),
// kein API-Call.
function LanguageSwitcher() {
  const { locale, setLocale } = useLocale();

  function select(next: Locale) {
    if (next !== locale) setLocale(next);
  }

  return (
    <div style={langSwitcherRow}>
      <button type="button" onClick={() => select("de")} style={langButton(locale === "de")}>
        DE
      </button>
      <button type="button" onClick={() => select("en")} style={langButton(locale === "en")}>
        EN
      </button>
    </div>
  );
}

function Header({ variant = "public" }: HeaderProps) {
  // Login/Logout im selben Tab crossen immer eine Layout-Grenze (Remount von
  // Header), daher wird ein einmaliges Lesen hier nur cross-tab relevant
  // (Tab A auf öffentlicher Seite, Tab B loggt sich ein/aus). Fix: auf
  // app:login/app:logout/storage hören statt nur einmal beim Render zu lesen.
  const [hasToken, setHasToken] = useState(() => Boolean(localStorage.getItem("access_token")));
  useEffect(() => {
    const sync = () => setHasToken(Boolean(localStorage.getItem("access_token")));
    window.addEventListener("app:login", sync);
    window.addEventListener("app:logout", sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener("app:login", sync);
      window.removeEventListener("app:logout", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 100,
        padding: "18px 24px",
      }}
    >
      <div
        style={{
          maxWidth: "1280px",
          margin: "0 auto",
          display: "flex",
          flexWrap: "wrap",
          rowGap: "10px",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 14px",
          borderRadius: theme.radius.pill,
          background: theme.glass.subtle.background,
          border: `1px solid ${theme.glass.subtle.border}`,
          backdropFilter: `blur(${theme.glass.subtle.blur})`,
          WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
        }}
      >
        <Link
          to={hasToken ? "/app/dashboard" : "/landing"}
          style={{
            textDecoration: "none",
            color: theme.colors.textPrimary,
            fontWeight: 900,
            fontSize: "1.5rem",
            letterSpacing: "-0.04em",
            padding: "0 10px",
          }}
        >
          ComAnalysis
        </Link>

        {/* Navigation */}
        <nav
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "4px",
            alignItems: "center",
            padding: "4px",
            borderRadius: theme.radius.pill,
            background: "transparent",
          }}
        >
          <LanguageSwitcher />

          {/* PUBLIC (logged in) */}
          {variant === "public" && hasToken && (
            <Link to="/app/dashboard" style={pillCta}>
              Zum Dashboard
            </Link>
          )}

          {/* PUBLIC (logged out) + AUTH */}
          {(variant === "auth" || (variant === "public" && !hasToken)) && (
            <>
              {variant === "public" ? (
                <Link to="/pricing" style={pillLinkGhost}>
                  Preise
                </Link>
              ) : null}

              <Link to="/login" style={pillLinkGhost}>
                Login
              </Link>

              <Link to="/register?src=header" style={pillCta}>
                Registrieren
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}

/* STYLES */

// display:inline-flex + minHeight statt nur Padding: garantiert die
// ~44px-Touch-Zielfläche unabhängig vom Zeilenumbruch-Verhalten des <a>
// (RESPONSIVE.md R-P1-2).
const pillLinkGhost: React.CSSProperties = {
  textDecoration: "none",
  color: theme.colors.textSecondary,
  fontWeight: 600,
  padding: "10px 16px",
  minHeight: "44px",
  boxSizing: "border-box",
  display: "inline-flex",
  alignItems: "center",
  borderRadius: theme.radius.pill,
};

const langSwitcherRow: React.CSSProperties = {
  display: "flex",
  gap: "2px",
  padding: "3px",
  borderRadius: theme.radius.pill,
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.border}`,
};

function langButton(active: boolean): React.CSSProperties {
  return {
    padding: "8px 10px",
    minHeight: "36px",
    boxSizing: "border-box",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: theme.radius.pill,
    border: "none",
    cursor: "pointer",
    fontWeight: 700,
    fontSize: "0.78rem",
    letterSpacing: "0.02em",
    background: active ? theme.gradients.ctaPrimary : "transparent",
    color: active ? theme.colors.bgDeep : theme.colors.textSecondary,
  };
}

const pillCta: React.CSSProperties = {
  textDecoration: "none",
  color: theme.colors.bgDeep,
  background: theme.gradients.ctaPrimary,
  fontWeight: 700,
  padding: "10px 18px",
  minHeight: "44px",
  boxSizing: "border-box",
  display: "inline-flex",
  alignItems: "center",
  borderRadius: theme.radius.pill,
  boxShadow: "0 10px 24px rgba(0, 0, 0, 0.35)",
};

export default Header;
