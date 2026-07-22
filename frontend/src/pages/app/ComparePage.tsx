import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GitCompare, Plus, Trash2 } from "lucide-react";
import { Button, Card } from "../../components/ui";
import { theme } from "../../components/ui/theme";
import FrequencyToggle from "../../components/analysis/FrequencyToggle";
import MetricCatalogPicker from "../../components/customAnalysis/MetricCatalogPicker";
import MultiLayerChart, { type ChartLayer } from "../../components/charts/MultiLayerChart";
import TimeRangeFilter from "../../components/charts/TimeRangeFilter";
import ChartTypeSelector from "../../components/charts/ChartTypeSelector";
import PercentChangeBadge from "../../components/charts/PercentChangeBadge";
import PriceComparisonSection from "../../components/charts/PriceComparisonSection";
import {
  computePercentChange,
  filterChartLayers,
  isPercentChangeEligibleUnit,
  mergeLayers,
  supportedChartTypes,
  type ChartType,
  type TimeRange,
} from "../../components/charts/chartUtils";
import ComparePivotTable from "../../components/compare/ComparePivotTable";
import SymbolSuggestField from "../../components/compare/SymbolSuggestField";
import { useCompare } from "../../hooks/useCompareContext";
import { useLivePrice } from "../../hooks/useLivePrice";
import { getCompanyColor, mapCompanyComplexMetrics, mapCompanyMetricsToLayers } from "../../compare/mapping";
import CrvTargetPanel from "../../components/metrics/CrvTargetPanel";
import { getCustomMetricsCatalog } from "../../api/customAnalysis";
import type { MetricCatalogEntry } from "../../types/customAnalysis";
import type { CompareGroupMeta, CompareLayer } from "../../types/compare";
import { getMetricConfig } from "../../config/metricsConfig";

// EVOLVING.md EV-041: Fundamentaldaten sind jährlich/quartalsweise - 1M-6M
// waeren dort fachlich sinnlos (keine Datenpunkte in diesem Fenster), daher
// nur die groben Ranges anbieten. Default bleibt "max" (D6) = heutiges
// Verhalten ohne Nutzerinteraktion.
const FUNDAMENTAL_RANGE_OPTIONS: TimeRange[] = ["1y", "2y", "5y", "max"];

type DraftRow = { id: number; value: string };

type LivePriceResult = { price: number | null; error: string | null };

/** Invisible per-symbol price poller — lets the page poll an arbitrary,
 * dynamic list of companies' live prices without violating the rules of
 * hooks (one mounted instance per symbol, each calling the existing
 * `useLivePrice` hook exactly once). Reports updates up via `onUpdate`
 * instead of rendering anything itself. */
function CompanyPriceSync({ symbol, onUpdate }: { symbol: string; onUpdate: (symbol: string, result: LivePriceResult) => void }) {
  const { price, error } = useLivePrice(symbol);
  useEffect(() => {
    onUpdate(symbol, { price, error });
  }, [symbol, price, error, onUpdate]);
  return null;
}

function ComparePage() {
  const {
    metrics,
    companies,
    frequency,
    hasStarted,
    setMetrics,
    setFrequency,
    addCompany,
    removeCompany,
    startComparison,
    clearAll,
  } = useCompare();

  const [catalog, setCatalog] = useState<MetricCatalogEntry[]>([]);
  const [isLoadingCatalog, setIsLoadingCatalog] = useState(true);
  const [catalogError, setCatalogError] = useState(false);
  const [catalogRetryKey, setCatalogRetryKey] = useState(0);
  const [livePrices, setLivePrices] = useState<Record<string, LivePriceResult>>({});
  // Zeitraum je Chart-Sektion (Schlüssel = metricKey) - React-State pro
  // Chart, keine URL-Persistenz (D7). Default "max" wird nicht hier,
  // sondern beim Lesen (`timeRanges[key] ?? "max"`) angewendet, damit ein
  // fehlender Eintrag exakt dem Ist-Zustand vor EV-041 entspricht.
  const [timeRanges, setTimeRanges] = useState<Record<string, TimeRange>>({});
  // EVOLVING.md CHART-006: analog timeRanges - Chart-Darstellung je Metrik-
  // Chart, Schlüssel = metricKey, nur React-State, keine Persistenz.
  const [chartTypes, setChartTypes] = useState<Record<string, ChartType>>({});

  const handlePriceUpdate = useCallback((symbol: string, result: LivePriceResult) => {
    setLivePrices((prev) => {
      const existing = prev[symbol];
      if (existing && existing.price === result.price && existing.error === result.error) return prev;
      return { ...prev, [symbol]: result };
    });
  }, []);

  const nextDraftId = useRef(0);
  function newDraft(): DraftRow {
    return { id: nextDraftId.current++, value: "" };
  }
  // Only seed the two empty input rows on a truly fresh workspace — if
  // companies were already restored from localStorage, showing extra blank
  // rows on every reload would duplicate what "+ Weiteres Unternehmen
  // hinzufügen" is for. Fixed negative ids instead of calling newDraft()
  // here (LAUNCH_AUDIT.md P2-10, react-hooks/refs) - reading nextDraftId.current
  // inside a useState lazy initializer is flagged as a ref access during
  // render; -1/-2 can never collide with the ref-based ids newDraft()
  // produces later (0, 1, 2, ...).
  const [draftRows, setDraftRows] = useState<DraftRow[]>(() =>
    companies.length === 0 ? [{ id: -1, value: "" }, { id: -2, value: "" }] : []
  );

  // MetricCatalogPicker only seeds its internal selection from
  // `initialMetrics` once at mount — bumping this key forces a remount so
  // "Alles löschen" actually clears the picker's checkboxes too, instead of
  // just the underlying `metrics` state the picker no longer reflects.
  const [pickerResetKey, setPickerResetKey] = useState(0);

  function handleClearAll() {
    clearAll();
    setPickerResetKey((key) => key + 1);
    setDraftRows([newDraft(), newDraft()]);
  }

  // Deselects every chosen metric without touching the companies — separate
  // from handleClearAll, which resets the whole workspace.
  function handleClearMetrics() {
    setMetrics([]);
    setPickerResetKey((key) => key + 1);
  }

  useEffect(() => {
    let isMounted = true;
    // Klassisches Loading-Flag vor einem Fetch - legitimer Effect-Zweck
    // (LAUNCH_AUDIT.md P2-10).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setIsLoadingCatalog(true);
    setCatalogError(false);
    getCustomMetricsCatalog()
      .then((data) => {
        if (isMounted) setCatalog(data);
      })
      .catch(() => {
        // Vorher: leerer Katalog ohne Erklärung - sah aus wie "keine
        // Kennzahlen verfügbar" statt "Abruf fehlgeschlagen". Jetzt eigener
        // Fehlerzustand mit Retry (siehe Render unten), catalogRetryKey
        // stößt diesen Effect erneut an.
        if (isMounted) {
          setCatalog([]);
          setCatalogError(true);
        }
      })
      .finally(() => {
        if (isMounted) setIsLoadingCatalog(false);
      });
    return () => {
      isMounted = false;
    };
  }, [catalogRetryKey]);

  function commitDraft(draftId: number, value: string) {
    const cleanSymbol = value.trim();
    if (cleanSymbol) {
      addCompany(cleanSymbol);
    }
    setDraftRows((prev) => prev.filter((row) => row.id !== draftId));
  }

  function updateDraftValue(draftId: number, value: string) {
    setDraftRows((prev) => prev.map((row) => (row.id === draftId ? { ...row, value: value.toUpperCase() } : row)));
  }

  function addDraftRow() {
    setDraftRows((prev) => [...prev, newDraft()]);
  }

  function removeDraftRow(draftId: number) {
    setDraftRows((prev) => prev.filter((row) => row.id !== draftId));
  }

  const doneCompanies = useMemo(() => companies.filter((c) => c.status === "done"), [companies]);

  const { layers, groups } = useMemo(() => {
    const selectedKeys = new Set(metrics.map((m) => m.key));

    let allLayers: CompareLayer[] = [];
    doneCompanies.forEach((company, index) => {
      const filteredMetrics = Object.fromEntries(
        Object.entries(company.metrics).filter(([key]) => selectedKeys.has(key))
      );
      const newLayers = mapCompanyMetricsToLayers(
        company.symbol,
        filteredMetrics,
        catalog,
        getCompanyColor(index),
        company.reporting_currency
      );
      allLayers = [...allLayers, ...newLayers];
    });

    const groupMeta: CompareGroupMeta[] = doneCompanies.map((c) => ({ groupId: c.symbol, groupLabel: c.symbol }));
    return { layers: allLayers, groups: groupMeta };
  }, [doneCompanies, catalog, metrics]);

  // The stock price is always shown as the first table row, independent of
  // the selected metrics — built separately from `layers` and prepended, so
  // it doesn't need a catalog entry and never enters the chart (it's a
  // live spot value, not a time series).
  const priceLayers: CompareLayer[] = useMemo(
    () =>
      doneCompanies.map((company, index) => {
        const live = livePrices[company.symbol];
        return {
          id: `${company.symbol}:__price`,
          groupId: company.symbol,
          groupLabel: company.symbol,
          metricKey: "__price",
          label: "Aktienkurs",
          axis: "left",
          color: getCompanyColor(index),
          chartEligible: false,
          value: live?.price ?? null,
          error: live?.error ?? null,
          // Kursbasiert, immer USD (NYSE/NASDAQ-Universum) - unabhängig von
          // der Berichtswährung der Fundamentaldaten (EVOLVING.md EV-022).
          currency: "USD",
        };
      }),
    [doneCompanies, livePrices]
  );

  // Chart-eligible metrics (time series) get their own chart box below and
  // have no single scalar `value` to show in a cell anyway — keeping them
  // out of the table avoids rows that are just "—" padding out the page.
  const tableLayers = useMemo(
    () => [...priceLayers, ...layers.filter((layer) => !layer.chartEligible)],
    [priceLayers, layers]
  );

  const chartGroups = useMemo(() => {
    const order: string[] = [];
    const byMetric = new Map<string, { metricKey: string; label: string; layers: ChartLayer[] }>();

    for (const layer of layers) {
      if (!layer.chartEligible || !layer.data) continue;
      if (!byMetric.has(layer.metricKey)) {
        byMetric.set(layer.metricKey, { metricKey: layer.metricKey, label: layer.label, layers: [] });
        order.push(layer.metricKey);
      }
      // EVOLVING.md EV-023: Currency nur für unit==="currency"-Kennzahlen
      // durchreichen - Ratio-/Margen-Charts (z. B. EV/EBIT) bekommen nie
      // eine Währungskennzeichnung, auch wenn layer.currency (aus EV-022,
      // dort unbedingt für die Pivot-Tabellenformatierung gesetzt) technisch
      // vorhanden ist.
      const isCurrencyMetric = getMetricConfig(layer.metricKey)?.unit === "currency";
      byMetric.get(layer.metricKey)!.layers.push({
        id: layer.id,
        label: layer.groupLabel,
        data: layer.data,
        axis: "left",
        color: layer.color,
        currency: isCurrencyMetric ? layer.currency : undefined,
      });
    }

    return order.map((metricKey) => byMetric.get(metricKey)!);
  }, [layers]);

  const complexGroups = useMemo(() => {
    const selectedKeys = new Set(metrics.map((m) => m.key));
    const order: string[] = [];
    const byMetric = new Map<string, { metricKey: string; label: string; results: ReturnType<typeof mapCompanyComplexMetrics> }>();

    for (const company of doneCompanies) {
      const filteredMetrics = Object.fromEntries(
        Object.entries(company.metrics).filter(([key]) => selectedKeys.has(key))
      );
      for (const result of mapCompanyComplexMetrics(company.symbol, filteredMetrics, catalog)) {
        if (!byMetric.has(result.metricKey)) {
          byMetric.set(result.metricKey, { metricKey: result.metricKey, label: result.label, results: [] });
          order.push(result.metricKey);
        }
        byMetric.get(result.metricKey)!.results.push(result);
      }
    }

    return order.map((metricKey) => byMetric.get(metricKey)!);
  }, [doneCompanies, catalog, metrics]);

  const hasAnything = metrics.length > 0 || companies.length > 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px", padding: "10px 6px 20px" }}>
      {doneCompanies.map((company) => (
        <CompanyPriceSync key={company.symbol} symbol={company.symbol} onUpdate={handlePriceUpdate} />
      ))}

      <section style={heroSection}>
        <div style={heroBadge}>Vergleich</div>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", flexWrap: "wrap", gap: "16px" }}>
          <h1 style={titleStyle}>Kennzahlen über beliebig viele Unternehmen vergleichen</h1>
          {hasAnything ? (
            <Button variant="ghost" onClick={handleClearAll} style={{ display: "inline-flex", alignItems: "center", gap: "8px" }}>
              <Trash2 size={15} />
              Alles löschen
            </Button>
          ) : null}
        </div>
        <p style={subtitleStyle}>
          Wähle beliebig viele Kennzahlen und füge so viele Unternehmen hinzu, wie du vergleichen möchtest —
          klicke dann auf „Vergleich starten“. Jedes danach neu hinzugefügte Unternehmen oder jede weitere
          Kennzahl wird automatisch mit abgerufen. Zeitserien-Kennzahlen erscheinen zusätzlich übereinander im
          Chart.
        </p>
      </section>

      <section style={resultsSection} data-tour="compare-metrics-picker">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "16px", marginBottom: "18px" }}>
          <div style={{ ...sectionEyebrow, marginBottom: 0 }}>Kennzahlen</div>
          {metrics.length > 0 ? (
            <Button variant="ghost" onClick={handleClearMetrics} style={{ display: "inline-flex", alignItems: "center", gap: "8px" }}>
              <Trash2 size={15} />
              Kennzahlen-Auswahl zurücksetzen
            </Button>
          ) : null}
        </div>
        {catalogError ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-start",
              gap: "10px",
              padding: "16px",
              borderRadius: theme.radius.md,
              border: `1px solid ${theme.colors.borderSubtle}`,
              background: theme.colors.panelAlt,
            }}
          >
            <span style={{ color: theme.colors.dangerText }}>
              Kennzahlen-Katalog konnte nicht geladen werden.
            </span>
            <Button variant="ghost" onClick={() => setCatalogRetryKey((key) => key + 1)}>
              Erneut versuchen
            </Button>
          </div>
        ) : (
          <MetricCatalogPicker
            key={pickerResetKey}
            catalog={catalog}
            isLoadingCatalog={isLoadingCatalog}
            initialMetrics={metrics}
            onChange={setMetrics}
            hideCriterion
          />
        )}
      </section>

      <section style={resultsSection}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "16px", marginBottom: "18px" }}>
          <div style={sectionEyebrow}>Unternehmen</div>
          <FrequencyToggle value={frequency} onChange={setFrequency} />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }} data-tour="compare-symbol-input">
          {companies.map((company) => (
            <div key={company.symbol} style={companyRowStyle}>
              <span style={companySymbolStyle}>{company.symbol}</span>
              <span style={statusStyle(!hasStarted && company.status === "running" ? "waiting" : company.status)}>
                {!hasStarted && company.status === "running"
                  ? "Wartet auf Start"
                  : company.status === "running"
                    ? "Läuft..."
                    : company.status === "error"
                      ? company.error ?? "Fehler"
                      : "Fertig"}
              </span>
              <button
                type="button"
                onClick={() => removeCompany(company.symbol)}
                aria-label="Unternehmen entfernen"
                style={iconButtonStyle}
              >
                <Trash2 size={14} color={theme.colors.textMuted} />
              </button>
            </div>
          ))}

          {draftRows.map((draft) => (
            <div key={draft.id} style={companyRowStyle}>
              <SymbolSuggestField
                value={draft.value}
                onChange={(value) => updateDraftValue(draft.id, value)}
                onCommit={(value) => commitDraft(draft.id, value)}
                placeholder="z. B. AAPL"
              />
              <button
                type="button"
                onClick={() => removeDraftRow(draft.id)}
                aria-label="Eingabe entfernen"
                style={iconButtonStyle}
              >
                <Trash2 size={14} color={theme.colors.textMuted} />
              </button>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px", marginTop: "12px" }}>
          <Button variant="ghost" onClick={addDraftRow} style={{ display: "inline-flex", alignItems: "center", gap: "8px" }}>
            <Plus size={15} />
            Weiteres Unternehmen hinzufügen
          </Button>

          {!hasStarted && metrics.length > 0 && companies.length > 0 ? (
            <Button
              variant="cta"
              onClick={startComparison}
              style={{ display: "inline-flex", alignItems: "center", gap: "8px" }}
            >
              <GitCompare size={15} />
              Vergleich starten
            </Button>
          ) : null}
        </div>
      </section>

      {tableLayers.length > 0 || complexGroups.length > 0 ? (
        <>
          <PriceComparisonSection symbols={doneCompanies.map((company) => company.symbol)} />

          {chartGroups.map((group) => {
            const range = timeRanges[group.metricKey] ?? "max";
            const bucketMode = frequency === "quarterly" ? "quarter" : "year";
            const filteredLayers = filterChartLayers(group.layers, range);
            const mergedRows = mergeLayers(filteredLayers, bucketMode);
            const hasEnoughData = mergedRows.length >= 2;
            // EVOLVING.md EV-051: dieselbe Bedingung wie die Currency-
            // Kennzeichnung aus EV-023 - Ratio-/Margen-Charts bekommen nie
            // eine %-Badge.
            const showPercentBadges = isPercentChangeEligibleUnit(getMetricConfig(group.metricKey)?.unit);
            // EVOLVING.md CHART-006: Säulen nur bis 4 Firmen gleichzeitig
            // (Betreiberentscheidung, testweise von 3 auf 4 angehoben) -
            // supportedChartTypes fällt darüber automatisch auf ["line"]
            // zurück, wodurch ChartTypeSelector sich selbst ausblendet
            // (options.length < 2).
            const chartTypeOptions = supportedChartTypes({
              unit: getMetricConfig(group.metricKey)?.unit,
              bucketMode,
              companyCount: filteredLayers.length,
              seriesLength: mergedRows.length,
            });
            const requestedChartType = chartTypes[group.metricKey] ?? "line";
            // Fängt den Fall ab, dass eine 4. Firma hinzukommt, während
            // dieser Chart bereits auf "bar" stand - Säulen werden dann
            // sofort erzwungen zurück auf Linie, nicht erst beim nächsten
            // manuellen Wechsel.
            const chartType = chartTypeOptions.includes(requestedChartType) ? requestedChartType : "line";

            return (
              <section key={group.metricKey} style={resultsSection}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px", marginBottom: "18px" }}>
                  <div style={{ ...sectionEyebrow, marginBottom: 0 }}>{group.label}</div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                    <TimeRangeFilter
                      value={range}
                      onChange={(next) => setTimeRanges((prev) => ({ ...prev, [group.metricKey]: next }))}
                      options={FUNDAMENTAL_RANGE_OPTIONS}
                    />
                    <ChartTypeSelector
                      value={chartType}
                      onChange={(next) => setChartTypes((prev) => ({ ...prev, [group.metricKey]: next }))}
                      options={chartTypeOptions}
                    />
                  </div>
                </div>
                {showPercentBadges ? (
                  <div style={percentBadgeRowStyle}>
                    {filteredLayers.map((layer) => (
                      <PercentChangeBadge key={layer.id} result={computePercentChange(layer.data)} color={layer.color} />
                    ))}
                  </div>
                ) : null}
                {hasEnoughData ? (
                  <MultiLayerChart layers={filteredLayers} height={300} bucketMode={bucketMode} chartType={chartType} />
                ) : (
                  <div style={emptyRangeStyle}>Für diesen Zeitraum liegen zu wenige Datenpunkte vor – Zeitraum vergrößern.</div>
                )}
              </section>
            );
          })}

          {complexGroups.map((group) => (
            <section key={group.metricKey} style={resultsSection}>
              <div style={sectionEyebrow}>{group.label}</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "16px" }}>
                {group.results.map((result) => (
                  <div key={result.symbol}>
                    <div style={{ ...companySymbolStyle, marginBottom: "10px" }}>{result.symbol}</div>
                    {result.error ? (
                      <div style={{ color: theme.colors.dangerText, fontSize: "0.88rem", lineHeight: 1.6 }}>{result.error}</div>
                    ) : (
                      <CrvTargetPanel value={result.value} />
                    )}
                  </div>
                ))}
              </div>
            </section>
          ))}

          {tableLayers.length > 0 ? (
            <section style={resultsSection}>
              <div style={sectionEyebrow}>Tabelle</div>
              <ComparePivotTable layers={tableLayers} groups={groups} />
            </section>
          ) : null}
        </>
      ) : (
        <Card variant="glass" style={{ textAlign: "center", padding: "48px 34px" }}>
          <GitCompare size={32} color={theme.colors.chrome} style={{ marginBottom: "16px" }} />
          <h2 style={{ margin: "0 0 10px 0", color: theme.colors.textPrimary, fontSize: "1.4rem" }}>
            Noch keine Ergebnisse
          </h2>
          <p style={{ margin: 0, color: theme.colors.textSecondary, fontSize: "0.98rem", lineHeight: 1.7, maxWidth: "560px", marginLeft: "auto", marginRight: "auto" }}>
            {hasStarted
              ? "Chart und Vergleichstabelle erscheinen automatisch, sobald die ersten Werte abgerufen wurden."
              : "Wähle oben mindestens eine Kennzahl, trage zwei oder mehr Unternehmen ein und klicke auf „Vergleich starten“."}
          </p>
        </Card>
      )}
    </div>
  );
}

const heroSection: React.CSSProperties = {
  background: theme.glass.elevated.background,
  backdropFilter: `blur(${theme.glass.elevated.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
  borderRadius: theme.radius.lg,
  padding: "34px 34px 36px",
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
};

const heroBadge: React.CSSProperties = {
  display: "inline-block",
  marginBottom: "16px",
  padding: "8px 12px",
  borderRadius: theme.radius.pill,
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.chrome,
  fontSize: "0.86rem",
  fontWeight: 700,
  letterSpacing: "0.03em",
};

const titleStyle: React.CSSProperties = {
  margin: "0 0 14px 0",
  // clamp() statt fix 2.4rem: skaliert auf sehr schmalen Screens herunter
  // statt die Zeile umzubrechen (RESPONSIVE.md R-P2-2).
  fontSize: "clamp(1.7rem, 5.5vw, 2.4rem)",
  lineHeight: 1.1,
  letterSpacing: "-0.04em",
  color: theme.colors.textPrimary,
};

const subtitleStyle: React.CSSProperties = {
  margin: 0,
  maxWidth: "860px",
  color: theme.colors.textPrimary,
  fontSize: "1.05rem",
  lineHeight: 1.8,
};

const resultsSection: React.CSSProperties = {
  background: theme.colors.panelAlt,
  borderRadius: theme.radius.lg,
  padding: "28px 30px",
  border: `1px solid ${theme.colors.border}`,
  boxShadow: "0 20px 50px rgba(0, 0, 0, 0.35)",
};

const emptyRangeStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  height: "160px",
  color: theme.colors.textMuted,
  fontSize: "0.92rem",
  border: `1px dashed ${theme.colors.borderSubtle}`,
  borderRadius: theme.radius.md,
};

const percentBadgeRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: "8px",
  marginBottom: "14px",
};

const sectionEyebrow: React.CSSProperties = {
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase",
  marginBottom: "18px",
};

const companyRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "10px",
  padding: "8px 12px",
  borderRadius: theme.radius.md,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
};

const companySymbolStyle: React.CSSProperties = {
  flex: 1,
  fontSize: "0.92rem",
  fontWeight: 700,
  color: theme.colors.textPrimary,
};

const iconButtonStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  border: "none",
  background: "transparent",
  cursor: "pointer",
  padding: "4px",
  flexShrink: 0,
};

function statusStyle(status: "running" | "done" | "error" | "waiting"): React.CSSProperties {
  return {
    fontSize: "0.82rem",
    fontWeight: 700,
    color:
      status === "error"
        ? theme.colors.dangerText
        : status === "running" || status === "waiting"
          ? theme.colors.textMuted
          : theme.colors.success,
  };
}

export default ComparePage;
