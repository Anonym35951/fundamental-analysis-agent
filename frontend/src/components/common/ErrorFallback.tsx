import { useTranslation } from "../../i18n/useTranslation";

// Eigene Datei statt in ErrorBoundary.tsx: eine Klassenkomponente kann keine
// Hooks (useTranslation) nutzen, und react-refresh/only-export-components
// verlangt, dass eine Datei mit einer Komponente nicht zusätzlich eine
// Klasse exportiert (EVOLVING.md § Internationalisierung, I18N-007).
export default function ErrorFallback({ onReload }: { onReload: () => void }) {
  const { t } = useTranslation("common");

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "60dvh",
        padding: "48px 24px",
        textAlign: "center",
        gap: "16px",
      }}
    >
      <h2 style={{ margin: 0, fontSize: "1.5rem" }}>
        {t("errorBoundary.title")}
      </h2>
      <p style={{ margin: 0, opacity: 0.8, maxWidth: "480px" }}>
        {t("errorBoundary.description")}
      </p>
      <button
        onClick={onReload}
        style={{
          padding: "10px 20px",
          borderRadius: "10px",
          border: "1px solid rgba(148, 163, 184, 0.4)",
          background: "rgba(59, 130, 246, 0.15)",
          color: "inherit",
          cursor: "pointer",
          fontSize: "0.95rem",
        }}
      >
        {t("errorBoundary.reloadButton")}
      </button>
    </div>
  );
}
