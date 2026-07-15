import Badge from "../ui/Badge";
import { formatPercentChange, type PercentChangeResult } from "./chartUtils";

type Props = {
  result: PercentChangeResult;
  /** Serienfarbe im Vergleichschart (EVOLVING.md EV-051) - zeigt einen
   * farbigen Punkt vor der Badge, damit sie derselben Firma wie ihre
   * Chart-Linie zugeordnet werden kann. Im Einzelchart weggelassen. */
  color?: string;
};

const REASON_LABELS: Record<Extract<PercentChangeResult, { percent: null }>["reason"], string> = {
  insufficient: "Zu wenig Datenpunkte im gewählten Zeitraum.",
  "zero-start": "Startwert ist 0 - prozentuale Veränderung nicht berechenbar.",
  "negative-start": "Startwert ist negativ - prozentuale Veränderung wäre irreführend.",
};

/** Reine Anzeige-Komponente (EVOLVING.md EV-051): grün bei positiver,
 * rot bei negativer, neutral grau bei ±0 % oder nicht berechenbarer
 * Veränderung - keine weitere Bewertungs-/Pfeil-Semantik (Neutralitäts-
 * Leitplanke, Abschnitt 2). */
export default function PercentChangeBadge({ result, color }: Props) {
  const tone = result.percent === null ? "neutral" : result.percent > 0 ? "success" : result.percent < 0 ? "danger" : "neutral";
  const title = result.percent === null ? REASON_LABELS[result.reason] : undefined;

  return (
    <Badge tone={tone} title={title} style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
      {color ? <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: color, flexShrink: 0 }} /> : null}
      {formatPercentChange(result)}
    </Badge>
  );
}
