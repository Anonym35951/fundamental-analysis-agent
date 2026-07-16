import { useEffect, useState } from "react";
import { ShieldCheck } from "lucide-react";
import { getDataSourceSummary, type DataSourceSummary } from "../../api/dataSource";
import { theme } from "../ui/theme";
import type { Frequency } from "../../types/frequency";

type Props = {
  symbol: string;
  frequency?: Frequency;
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
    // Alten Badge-Stand sofort löschen, bevor der neue Fetch für ein
    // gewechseltes Symbol/Frequenz startet - sonst würde kurzzeitig die
    // Quelle des VORHERIGEN Symbols neben dem neuen Analyseergebnis stehen
    // (LAUNCH_AUDIT.md P2-10, legitimer Reset-bei-Props-Wechsel-Fall).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSummary(null);

    if (!symbol) return undefined;

    // EV-134: /metrics/data-source kennt "ttm" nicht (kein Model.py-Metrik-
    // Endpoint, sondern ein bespoke DataLoader-Aufruf ohne ttm-Delegation) -
    // die Datenherkunft für ttm ist ohnehin identisch zur annual-Quelle
    // (beide SEC/Quartalsdaten), daher hier auf "annual" abbilden statt den
    // Badge für den neuen ttm-Default stillschweigend verschwinden zu lassen.
    const sourceFrequency = frequency === "ttm" ? "annual" : frequency;
    getDataSourceSummary(symbol, sourceFrequency)
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
