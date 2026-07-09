import { useEffect, useState } from "react";
import { ShieldCheck } from "lucide-react";
import { getDataSourceSummary, type DataSourceSummary } from "../../api/dataSource";
import { theme } from "../ui/theme";

type Props = {
  symbol: string;
  frequency?: "annual" | "quarterly";
};

function formatAsOf(asOf: string | null): string | null {
  if (!asOf) return null;
  const parsed = new Date(`${asOf}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit", year: "numeric" });
}

/** Zeigt Herkunft + Stand der Fundamentaldaten eines Symbols direkt neben dem
 * Analyseergebnis — das sichtbarste Vertrauenssignal fürs Transparenz-
 * Versprechen ("woher kommt diese Zahl, wie aktuell ist sie"). Fehlschläge
 * bleiben bewusst unauffällig (kein Fehlertext), analog zu LivePriceBadge,
 * da ein sekundäres Detail neben dem eigentlichen Ergebnis nicht mit einer
 * Fehlermeldung konkurrieren soll. */
export default function SourceBadge({ symbol, frequency = "annual" }: Props) {
  const [summary, setSummary] = useState<DataSourceSummary | null>(null);

  useEffect(() => {
    let isMounted = true;
    setSummary(null);

    if (!symbol) return undefined;

    getDataSourceSummary(symbol, frequency)
      .then((result) => {
        if (isMounted && !("error" in result)) setSummary(result);
      })
      .catch(() => {
        // Kein Badge statt Fehlermeldung — siehe Kommentar oben.
      });

    return () => {
      isMounted = false;
    };
  }, [symbol, frequency]);

  if (!summary) return null;

  const asOfLabel = formatAsOf(summary.as_of);

  return (
    <span
      title="Fundamentaldaten stammen direkt aus SEC-Filings, nicht aus Dritt-Aggregatoren."
      style={wrapper}
    >
      <ShieldCheck size={13} style={{ flexShrink: 0 }} />
      {summary.source}
      {asOfLabel ? <span style={separator}>·</span> : null}
      {asOfLabel ? <span>Stand {asOfLabel}</span> : null}
    </span>
  );
}

const wrapper = {
  display: "inline-flex",
  alignItems: "center",
  gap: "5px",
  padding: "4px 10px",
  borderRadius: theme.radius.pill,
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textSecondary,
  fontSize: "0.78rem",
  fontWeight: 700,
  whiteSpace: "nowrap" as const,
};

const separator = {
  color: theme.colors.textMuted,
};
