import InfoTooltip from "../ui/InfoTooltip";
import { theme } from "../ui/theme";
import { formatLabel, formatMetricValue } from "./metricFormatting";

type TargetFields = "WC" | "BUY" | "FV" | "SELL";

type CrvTargetPanelProps = {
  /** Raw value of one of: evaluate_tbv_bandwidth / evaluate_ebit_bandwidth
   * (bandwidth shape), calculate_crv (CRV shape), or
   * calculate_course_target_PriceMultiples / calculate_course_target_EVMultiples
   * (course-target shape). Shape is detected by duck-typing since these are
   * 4 structurally distinct dicts from agent/Model.py — see each render
   * helper below for the exact fields it expects. */
  value?: unknown;
};

const cardStyle = {
  padding: "16px",
  borderRadius: theme.radius.md,
  background: theme.colors.panel,
  border: `1px solid ${theme.colors.border}`,
};

const rowStyle = {
  display: "flex",
  justifyContent: "space-between",
  gap: "10px",
  padding: "8px 10px",
  borderRadius: "11px",
  background: theme.colors.panelAlt,
};

const sectionLabelStyle = {
  color: theme.colors.chrome,
  fontSize: "0.78rem",
  fontWeight: 800,
  textTransform: "uppercase" as const,
  marginBottom: "8px",
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasTargetFields(value: unknown): value is Record<TargetFields, number> {
  return isRecord(value) && ["WC", "BUY", "FV", "SELL"].every((key) => typeof value[key] === "number");
}

function Row({ label, value, formatKey }: { label: string; value: unknown; formatKey?: string }) {
  return (
    <div style={rowStyle}>
      <span style={{ display: "inline-flex", alignItems: "center", gap: "5px", color: theme.colors.chrome, fontSize: "0.78rem", fontWeight: 800 }}>
        {label}
        {formatKey ? <InfoTooltip metricKey={formatKey} /> : null}
      </span>
      <span style={{ color: theme.colors.textPrimary, fontSize: "0.8rem", fontWeight: 700 }}>
        {formatMetricValue(value, formatKey)}
      </span>
    </div>
  );
}

/** evaluate_tbv_bandwidth / evaluate_ebit_bandwidth: { targets: {WC,BUY,SELL},
 * current: {price, ...}, signal, message } — the historical "pb"/"ebit"
 * DataFrame field is deliberately ignored (not JSON-meaningful). */
function BandwidthCard({
  targets,
  current,
  signal,
  message,
}: {
  targets: Record<string, unknown>;
  current: Record<string, unknown>;
  signal?: unknown;
  message?: unknown;
}) {
  return (
    <div style={cardStyle}>
      {typeof signal === "string" ? (
        <div
          style={{
            display: "inline-block",
            marginBottom: "10px",
            padding: "4px 10px",
            borderRadius: theme.radius.pill,
            fontSize: "0.78rem",
            fontWeight: 800,
            textTransform: "uppercase",
            color: signal === "buy" ? theme.colors.successText : signal === "sell" ? theme.colors.dangerText : theme.colors.textSecondary,
            background: signal === "buy" ? theme.colors.successSoft : signal === "sell" ? theme.colors.dangerSoft : theme.colors.chromeSoft,
          }}
        >
          {signal}
        </div>
      ) : null}

      <div style={{ display: "flex", flexDirection: "column", gap: "7px", marginBottom: "12px" }}>
        {Object.entries(current)
          .filter(([, value]) => value !== null && value !== undefined)
          .map(([key, value]) => (
            <Row key={key} label={formatLabel(key)} value={value} formatKey={key} />
          ))}
      </div>

      <div style={sectionLabelStyle}>Kursziele</div>
      <div style={{ display: "flex", flexDirection: "column", gap: "7px" }}>
        {Object.entries(targets).map(([key, value]) => (
          <Row key={key} label={key} value={value} formatKey={key} />
        ))}
      </div>

      {typeof message === "string" ? (
        <div style={{ marginTop: "10px", color: theme.colors.textSecondary, fontSize: "0.82rem", lineHeight: 1.5 }}>
          {message}
        </div>
      ) : null}
    </div>
  );
}

/** calculate_crv: { crv_conservative, crv_aggressive, inputs: {...},
 * course_targets: {WC,BUY,FV,SELL} } — one multiple's CRV result. Used
 * standalone (custom-analysis catalog's single "CRV" metric) or as one
 * entry of the multi-multiple grid below (Vollanalyse). */
function CrvCard({ value, title }: { value: Record<string, unknown>; title?: string }) {
  const inputs = isRecord(value.inputs) ? value.inputs : {};
  const courseTargets = isRecord(value.course_targets) ? value.course_targets : {};

  return (
    <div style={cardStyle}>
      {title ? (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            color: theme.colors.textPrimary,
            fontSize: "0.9rem",
            fontWeight: 800,
            marginBottom: "10px",
          }}
        >
          {typeof value.crv_positive === "boolean" ? (
            <span style={{ color: value.crv_positive ? theme.colors.successText : theme.colors.dangerText, fontWeight: 900 }}>
              {value.crv_positive ? "✔" : "✘"}
            </span>
          ) : null}
          <span>{title}</span>
        </div>
      ) : null}

      {typeof value.error === "string" ? (
        <div style={{ color: theme.colors.dangerText, fontSize: "0.82rem", lineHeight: 1.5 }}>{value.error}</div>
      ) : (
        <>
          <div style={{ display: "flex", flexDirection: "column", gap: "7px", marginBottom: "12px" }}>
            {typeof value.crv_conservative === "number" ? (
              <Row label="CRV konservativ" value={value.crv_conservative} formatKey="crv_conservative" />
            ) : null}
            {typeof value.crv_aggressive === "number" ? (
              <Row label="CRV aggressiv" value={value.crv_aggressive} formatKey="crv_aggressive" />
            ) : null}
            {Object.entries(inputs).map(([key, val]) => (
              <Row key={key} label={formatLabel(key)} value={val} formatKey={key} />
            ))}
          </div>

          {Object.keys(courseTargets).length > 0 ? (
            <>
              <div style={sectionLabelStyle}>Kursziele</div>
              <div style={{ display: "flex", flexDirection: "column", gap: "7px" }}>
                {Object.entries(courseTargets).map(([key, val]) => (
                  <Row key={key} label={key} value={val} formatKey={key} />
                ))}
              </div>
            </>
          ) : null}
        </>
      )}
    </div>
  );
}

/** Vollanalyse CRV section (AgentAction's per-sector evaluate-CRV helper):
 * { multiples_used: string[], crv_results: Record<multiple, CrvCard-shape> }
 * — one CrvCard per multiple in a grid. */
function MultiCrvGrid({ multiplesUsed, crvResults }: { multiplesUsed?: unknown; crvResults: Record<string, unknown> }) {
  const entries = Object.entries(crvResults);
  if (entries.length === 0) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      {Array.isArray(multiplesUsed) && multiplesUsed.length > 0 ? (
        <div style={{ color: theme.colors.textMuted, fontSize: "0.84rem", lineHeight: 1.6 }}>
          Verwendete Multiples: {multiplesUsed.join(", ")}
        </div>
      ) : null}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "12px" }}>
        {entries.map(([multiple, data]) => (
          <CrvCard key={multiple} value={isRecord(data) ? data : {}} title={multiple} />
        ))}
      </div>
    </div>
  );
}

/** calculate_course_target_PriceMultiples / _EVMultiples, normalized to a
 * flat WC/BUY/FV/SELL price-target record by the caller below. */
function CourseTargetCard({ title, targets }: { title?: string; targets: Record<TargetFields, number> }) {
  return (
    <div style={cardStyle}>
      {title ? (
        <div style={{ color: theme.colors.textPrimary, fontSize: "0.9rem", fontWeight: 800, marginBottom: "10px" }}>
          {title}
        </div>
      ) : null}
      <div style={sectionLabelStyle}>Kursziele</div>
      <div style={{ display: "flex", flexDirection: "column", gap: "7px" }}>
        {Object.entries(targets).map(([key, val]) => (
          <Row key={key} label={key} value={val} formatKey={key} />
        ))}
      </div>
    </div>
  );
}

/** Dedicated renderer for CRV / bandwidth / course-target valuation
 * results, shared between the Analyze custom-analysis result list and the
 * Compare page's "Bewertung (komplex)" section. Detects which of the 4
 * structurally distinct agent/Model.py shapes it received and renders
 * accordingly. */
export default function CrvTargetPanel({ value }: CrvTargetPanelProps) {
  if (!isRecord(value)) return null;

  if (typeof value.error === "string") {
    return <div style={{ color: theme.colors.dangerText, fontSize: "0.88rem", lineHeight: 1.6 }}>{value.error}</div>;
  }

  if (isRecord(value.targets) && isRecord(value.current)) {
    return <BandwidthCard targets={value.targets} current={value.current} signal={value.signal} message={value.message} />;
  }

  if (isRecord(value.crv_results)) {
    return <MultiCrvGrid multiplesUsed={value.multiples_used} crvResults={value.crv_results} />;
  }

  if (typeof value.crv_conservative === "number" || typeof value.crv_aggressive === "number") {
    return <CrvCard value={value} />;
  }

  if (typeof value.worst_case_price === "number") {
    return (
      <CourseTargetCard
        targets={{
          WC: value.worst_case_price as number,
          BUY: value.buy_price as number,
          FV: value.fair_value_price as number,
          SELL: value.sell_price as number,
        }}
      />
    );
  }

  const evEntry = Object.entries(value).find(([, entryValue]) => hasTargetFields(entryValue));
  if (evEntry) {
    const [multipleName, targets] = evEntry;
    // hasTargetFields() already verified the shape above; Array.find()'s
    // type-predicate narrowing doesn't propagate through a destructured
    // tuple element, so TS still sees `targets` as unknown here.
    return <CourseTargetCard title={multipleName} targets={targets as Record<TargetFields, number>} />;
  }

  return null;
}
