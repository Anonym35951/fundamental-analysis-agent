import { useEffect, useState } from "react";
import { CheckCircle2, XCircle } from "lucide-react";
import { getDataSourceStatus, type DataSourceStatusEntry } from "../../api/dataSourceStatus";
import { theme } from "../ui/theme";

/** Kompakte Erreichbarkeits-Anzeige der externen Datenquellen ("SEC: ✓ ·
 * Yahoo: ✓") — passt zur Transparenz-Positionierung: bei einem Ausfall soll
 * sofort klar sein, dass es an der Datenquelle liegt und nicht an
 * ComAnalysis selbst. Fehlschläge beim Laden bleiben bewusst unauffällig
 * (kein Fehlertext für ein sekundäres Vertrauens-Widget), analog zu
 * SourceBadge/LivePriceBadge. */
export default function DataSourceStatusWidget() {
  const [sources, setSources] = useState<DataSourceStatusEntry[] | null>(null);

  useEffect(() => {
    let isMounted = true;

    getDataSourceStatus()
      .then((result) => {
        if (isMounted) setSources(result);
      })
      .catch(() => {
        // Kein Widget statt Fehlermeldung — siehe Kommentar oben.
      });

    return () => {
      isMounted = false;
    };
  }, []);

  if (!sources) return null;

  return (
    <div style={wrapper}>
      <span style={label}>Datenquellen</span>
      {sources.map((source) => (
        <span key={source.name} style={pill}>
          {source.status === "ok" ? (
            <CheckCircle2 size={13} color={theme.colors.successText} />
          ) : (
            <XCircle size={13} color={theme.colors.dangerText} />
          )}
          {source.name}
        </span>
      ))}
    </div>
  );
}

const wrapper: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  flexWrap: "wrap",
  gap: "8px",
  marginTop: "14px",
};

const label: React.CSSProperties = {
  color: theme.colors.textMuted,
  fontSize: "0.78rem",
  fontWeight: 700,
  textTransform: "uppercase",
  letterSpacing: "0.03em",
};

const pill: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: "5px",
  padding: "4px 10px",
  borderRadius: theme.radius.pill,
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textSecondary,
  fontSize: "0.8rem",
  fontWeight: 700,
};
