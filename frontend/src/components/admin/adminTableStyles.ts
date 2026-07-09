import type { CSSProperties } from "react";
import { theme } from "../ui/theme";

/** Gemeinsame Tabellen-/Panel-Styles fuer den Admin-Bereich (Uebersicht +
 * Kunden-Tab), extrahiert aus AdminDashboardPage.tsx, um Copy-Paste zu
 * vermeiden. */
export const panel: CSSProperties = {
  background: theme.glass.subtle.background,
  backdropFilter: `blur(${theme.glass.subtle.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
  borderRadius: theme.radius.lg,
  padding: "28px",
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textPrimary,
  boxSizing: "border-box",
};

export const panelTitle: CSSProperties = {
  fontSize: "1.2rem",
  fontWeight: 800,
  marginBottom: "18px",
  color: theme.colors.textPrimary,
};

export const table: CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: "0.9rem",
};

export const th: CSSProperties = {
  textAlign: "left",
  padding: "8px 10px",
  color: theme.colors.textSecondary,
  fontWeight: 700,
  fontSize: "0.78rem",
  textTransform: "uppercase",
  letterSpacing: "0.03em",
  borderBottom: `1px solid ${theme.glass.subtle.border}`,
};

export const td: CSSProperties = {
  padding: "8px 10px",
  color: theme.colors.textPrimary,
  borderBottom: `1px solid ${theme.glass.subtle.border}`,
};

export const emptyState: CSSProperties = {
  color: theme.colors.textSecondary,
  fontSize: "0.92rem",
  padding: "12px 0",
};

export const trClickable: CSSProperties = {
  cursor: "pointer",
};
