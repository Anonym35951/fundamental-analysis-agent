import type { TooltipContentProps } from "recharts";
import type { ValueType, NameType } from "recharts/types/component/DefaultTooltipContent";
import { theme } from "../ui/theme";
import { extractTooltipRows, formatCompactNumber, type ChartLayer } from "./chartUtils";

type Props = TooltipContentProps<ValueType, NameType> & {
  /** Alle ausgewählten Firmen-Layer (nicht nur die, für die Recharts an
   * dieser X-Position einen Rechart-`payload`-Eintrag gebaut hat) -
   * EVOLVING.md EV-031: iteriert bewusst über `layers`, nicht über
   * `payload`, damit jede Firma an jeder Hover-Position auftaucht, auch
   * wenn sie im gehoverten Bucket keinen Wert hat ("–" statt Verschwinden). */
  layers: ChartLayer[];
  /** Nur bei gemischten Originalwährungen (EVOLVING.md EV-023) true - dann
   * zeigt jede Zeile zusätzlich ihren ISO-Code, da die Werte sonst nicht
   * direkt vergleichbar wären. Im Standardfall (eine einheitliche Währung
   * oder gar keine bekannt) bleibt die Tooltip-Optik unverändert - die
   * einzige sichtbare Änderung ist dann die Chart-Unterzeile. */
  showCurrencyPerRow?: boolean;
};

/** Ersetzt Rechart's Standard-Tooltip (der nur Serien mit definiertem Wert
 * an der gehoverten X-Position auflistet - Root Cause des "nur 1-2 Firmen
 * im Tooltip"-Bugs, siehe EVOLVING.md Abschnitt 5.3/6 P5). Zusammen mit dem
 * Perioden-Bucketing aus EV-030 sind die meisten Zeilen jetzt zwar bereits
 * vollständig besetzt, aber nicht alle (kurze Firmenhistorie, fehlende
 * Meldung für ein Jahr) - dieser Tooltip zeigt für solche Fälle explizit
 * "–" statt die Firma stillschweigend wegzulassen. */
export default function ChartTooltip({ active, payload, layers, showCurrencyPerRow }: Props) {
  if (!active || !payload || payload.length === 0) return null;

  const row = payload[0].payload as Record<string, string | number> | undefined;
  if (!row) return null;

  const rows = extractTooltipRows(row, layers);

  return (
    <div style={containerStyle}>
      <p style={labelStyle}>{row.label}</p>
      {rows.map((tooltipRow) => (
        <div key={tooltipRow.id} style={rowStyle}>
          <span style={nameGroupStyle}>
            <span style={{ ...dotStyle, background: tooltipRow.color }} />
            <span style={nameStyle}>{tooltipRow.label}</span>
          </span>
          <span style={valueStyle}>
            {tooltipRow.value !== null
              ? `${formatCompactNumber(tooltipRow.value)}${
                  showCurrencyPerRow && tooltipRow.currency ? ` ${tooltipRow.currency}` : ""
                }`
              : "–"}
          </span>
        </div>
      ))}
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  margin: 0,
  padding: "10px 12px",
  background: theme.colors.panel,
  border: `1px solid ${theme.colors.border}`,
  borderRadius: theme.radius.sm,
  color: theme.colors.textPrimary,
  fontSize: "0.8rem",
  minWidth: "160px",
};

const labelStyle: React.CSSProperties = {
  margin: "0 0 6px 0",
  color: theme.colors.textMuted,
  fontWeight: 700,
};

const rowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: "10px",
  padding: "2px 0",
};

const nameGroupStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  minWidth: 0,
};

const dotStyle: React.CSSProperties = {
  width: "8px",
  height: "8px",
  borderRadius: "50%",
  flexShrink: 0,
  marginRight: "6px",
};

const nameStyle: React.CSSProperties = {
  color: theme.colors.textSecondary,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};

const valueStyle: React.CSSProperties = {
  color: theme.colors.textPrimary,
  fontWeight: 700,
  whiteSpace: "nowrap",
};
