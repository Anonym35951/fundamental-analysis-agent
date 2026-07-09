import { motion } from "framer-motion";
import { theme } from "../ui/theme";

type Props = {
  percent: number;
  status: string;
  currentStep?: string | null;
  /** How many discrete segments to render — purely visual, has no relation
   * to the underlying done/total counts beyond filling proportionally. */
  segments?: number;
};

/** System-status-style progress readout: a row of discrete chrome segments
 * fills left-to-right by percent, with an animated sweep highlight on the
 * leading edge — reads as an instrument panel readout rather than a loading
 * spinner. Purely presentational; callers keep owning all polling/state. */
export default function ProgressReadout({ percent, status, currentStep, segments = 24 }: Props) {
  const clamped = Math.max(0, Math.min(100, percent));
  const filledSegments = Math.round((clamped / 100) * segments);
  const isRunning = status === "running";

  return (
    <div style={{ width: "100%" }}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          marginBottom: "8px",
        }}
      >
        <span
          style={{
            fontSize: "0.74rem",
            fontWeight: 800,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            color: theme.colors.chrome,
          }}
        >
          {isRunning ? "Verarbeitung läuft" : status}
        </span>
        <span
          style={{
            fontSize: "0.82rem",
            fontWeight: 800,
            color: theme.colors.textPrimary,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {clamped}%
        </span>
      </div>

      <div style={{ display: "flex", gap: "3px" }}>
        {Array.from({ length: segments }).map((_, index) => {
          const isFilled = index < filledSegments;
          const isLeadingEdge = index === filledSegments - 1 && isRunning;

          return (
            <div
              key={index}
              style={{
                position: "relative",
                flex: 1,
                height: "5px",
                borderRadius: theme.radius.pill,
                overflow: "hidden",
                background: isFilled ? theme.colors.chromeStrong : theme.colors.borderSubtle,
                transition: `background ${theme.motion.base} ${theme.motion.easing}`,
              }}
            >
              {isLeadingEdge ? (
                <motion.div
                  initial={{ opacity: 0.4 }}
                  animate={{ opacity: [0.4, 1, 0.4] }}
                  transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: theme.gradients.cardSheen,
                  }}
                />
              ) : null}
            </div>
          );
        })}
      </div>

      {isRunning ? (
        <div
          style={{
            marginTop: "9px",
            fontSize: "0.8rem",
            color: theme.colors.textSecondary,
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          }}
        >
          {currentStep || "Verarbeitung läuft…"}
        </div>
      ) : null}
    </div>
  );
}
