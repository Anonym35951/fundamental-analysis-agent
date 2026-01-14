// frontend/src/components/ResultsView.tsx
import type { FullResult } from "../types/api";
import { highlight } from "../utils/highlight";
import "./results.css";

type Props = {
  data: FullResult;
  query?: string;
  onlyFailed?: boolean;
  onlyCrv?: boolean;
};

function splitKey(key: string) {
  const [name, freq] = key.split("|");
  return { name: name ?? key, freq: freq ?? "" };
}

function getIcon(meets: any) {
  if (meets === true) return "✔";
  if (meets === false) return "✘";
  return "•";
}

function formatVal(v: any) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number")
    return Number.isFinite(v) ? v.toFixed(4).replace(/\.?0+$/, "") : String(v);
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.join(", ");
  return String(v);
}

function CriterionRow({
  label,
  obj,
  query,
}: {
  label: string;
  obj: any;
  query?: string;
}) {
  const meets = obj?.meets_criterion;
  const icon = getIcon(meets);
  const value = obj?.value;

  const subEntries =
    obj && typeof obj === "object"
      ? Object.entries(obj).filter(
          ([k]) => !["meets_criterion", "value"].includes(k)
        )
      : [];

  return (
    <div className="crit">
      <div className="critTop">
        <div className="critLeft">
          <span
            className={`critIcon ${
              meets === true ? "ok" : meets === false ? "bad" : "neutral"
            }`}
          >
            {icon}
          </span>
          <span className="critLabel">
            {query ? highlight(label, query) : label}
          </span>
        </div>

        {value !== undefined && value !== null && (
          <div className="critValue">
            {query ? highlight(formatVal(value), query) : formatVal(value)}
          </div>
        )}
      </div>

      {subEntries.length > 0 && (
        <div className="critSubs">
          {subEntries.map(([k, v]) => (
            <div key={k} className="critSub">
              <span className="critArrow">↳</span>
              <span className="critSubKey">
                {query
                  ? highlight(k.replaceAll("_", " "), query)
                  : k.replaceAll("_", " ")}
                :
              </span>
              <span className="critSubVal">
                {query ? highlight(String(v), query) : String(v)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CRVBlock({ crv, query }: { crv: any; query?: string }) {
  if (!crv || typeof crv !== "object") return null;
  const crvResults = crv.crv_results ?? {};

  return (
    <div className="crv">
      <div className="crvTitle">📈 CRV – Kursziele</div>

      {Object.entries(crvResults).map(([multiple, data]: any) => {
        if (!data || typeof data !== "object") return null;

        if (data.error) {
          return (
            <div key={multiple} className="crvRow">
              <div className="crvHeader">
                <span className="bad">✘</span>{" "}
                {query ? highlight(multiple, query) : multiple}
              </div>
              <div className="crvError">
                Fehler:{" "}
                {query
                  ? highlight(String(data.error), query)
                  : String(data.error)}
              </div>
            </div>
          );
        }

        const positive = !!data.crv_positive;
        const targets = data.course_targets ?? data.targets ?? {};

        return (
          <div key={multiple} className="crvRow">
            <div className="crvHeader">
              <span className={positive ? "ok" : "bad"}>
                {positive ? "✔" : "✘"}
              </span>{" "}
              {query ? highlight(multiple, query) : multiple}
            </div>

            <div className="crvTargets">
              {Object.entries(targets).map(([k, v]) => (
                <div key={k} className="crvTarget">
                  <span className="crvK">{k}</span>
                  <span className="crvV">
                    {query ? highlight(String(v), query) : String(v)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function AnalysisCard({
  keyName,
  payload,
  query,
  onlyFailed,
  onlyCrv,
}: {
  keyName: string;
  payload: any;
  query?: string;
  onlyFailed?: boolean;
  onlyCrv?: boolean;
}) {
  const { name, freq } = splitKey(keyName);

  const keys =
    payload && typeof payload === "object"
      ? Object.keys(payload).filter(
          (k) =>
            ![
              "symbol",
              "frequency",
              "overall_assessment",
              "message",
              "crv",
            ].includes(k)
        )
      : [];

  const hasRenderableCriteria = keys.some(
    (k) => typeof payload[k] === "object"
  );

  // ✅ NUR Dividenden / Average Grower
  const isDividendAnalysis =
    name.toLowerCase().includes("dividenden") ||
    name.toLowerCase().includes("average");

  return (
    <details className="card" open>
      <summary className="cardSummary">
        <div className="cardTitle">
          {query ? highlight(name, query) : name}
          <span className="pill">{freq.toUpperCase()}</span>
        </div>
      </summary>

      {payload?.message && (
        <div className="cardMsg">
          {query ? highlight(payload.message, query) : payload.message}
        </div>
      )}

      <div className="cardBody">
        {!hasRenderableCriteria && !payload?.crv && isDividendAnalysis ? (
          <div style={{ fontSize: 13, opacity: 0.75, padding: "6px 2px" }}>
            💡 Dieses Unternehmen zahlt aktuell keine Dividende.
          </div>
        ) : (
          <>
            {!onlyCrv &&
              keys.map((k) => {
                const v = payload[k];
                const label = k
                  .replaceAll("_", " ")
                  .replace(/\b\w/g, (c) => c.toUpperCase());

                if (onlyFailed && v?.meets_criterion !== false) return null;

                return typeof v === "object" ? (
                  <CriterionRow
                    key={k}
                    label={label}
                    obj={v}
                    query={query}
                  />
                ) : null;
              })}

            {payload?.crv && <CRVBlock crv={payload.crv} query={query} />}
          </>
        )}
      </div>
    </details>
  );
}

export default function ResultsView({
  data,
  query = "",
  onlyFailed = false,
  onlyCrv = false,
}: Props) {
  const grouped: Record<string, { key: string; payload: any }[]> = {};

  for (const [k, payload] of Object.entries(data.results ?? {})) {
    const { name } = splitKey(k);
    if (!grouped[name]) grouped[name] = [];
    grouped[name].push({ key: k, payload });
  }

  return (
    <div className="results">
      <div className="resultsGrid">
        {Object.entries(grouped).map(([name, items]) => (
          <div key={name} className="group">
            <div className="groupTitle">
              {query ? highlight(name, query) : name}
            </div>
            <div className="groupList">
              {items.map(({ key, payload }) => (
                <AnalysisCard
                  key={key}
                  keyName={key}
                  payload={payload}
                  query={query}
                  onlyFailed={onlyFailed}
                  onlyCrv={onlyCrv}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}