import { describe, expect, it } from "vitest";
import {
  appendLivePoint,
  bucketKey,
  computePercentChange,
  computeStartEndTicks,
  extractTooltipRows,
  filterChartLayers,
  filterSeriesByRange,
  filterToLastDays,
  formatPercentChange,
  formatPriceTick,
  isPercentChangeEligibleUnit,
  layersCurrencyState,
  localIsoDate,
  mergeLayers,
  normalizePriceSeries,
  type ChartLayer,
  type TimeRange,
} from "./chartUtils";

describe("bucketKey (EVOLVING.md EV-030)", () => {
  it("reduces a date to its calendar year", () => {
    expect(bucketKey("2024-09-30", "year")).toBe("2024");
  });

  it("reduces a date to its calendar quarter", () => {
    expect(bucketKey("2024-01-15", "quarter")).toBe("Q1 2024");
    expect(bucketKey("2024-04-01", "quarter")).toBe("Q2 2024");
    expect(bucketKey("2024-09-30", "quarter")).toBe("Q3 2024");
    expect(bucketKey("2024-12-31", "quarter")).toBe("Q4 2024");
  });

  it("passes the date through unchanged in date mode", () => {
    expect(bucketKey("2024-09-30", "date")).toBe("2024-09-30");
  });
});

function layer(id: string, points: Array<[string, number]>): ChartLayer {
  return {
    id,
    label: id,
    axis: "left",
    color: "#000",
    data: points.map(([date, value]) => ({ date, value })),
  };
}

describe("mergeLayers (EVOLVING.md EV-030)", () => {
  it("merges two companies with offset fiscal dates into one row per year", () => {
    // Root Cause des Tooltip-Bugs: AAPL (FY-Ende Sept) und KO (FY-Ende Dez)
    // haben nie denselben exakten Datums-Schluessel - year-Bucketing muss
    // sie trotzdem in dieselbe Zeile zusammenfuehren.
    const layers = [
      layer("AAPL", [
        ["2023-09-30", 100],
        ["2024-09-30", 110],
      ]),
      layer("KO", [
        ["2023-12-31", 200],
        ["2024-12-31", 220],
      ]),
    ];

    const rows = mergeLayers(layers, "year");

    expect(rows).toHaveLength(2);
    const row2023 = rows.find((r) => r.label === "2023")!;
    const row2024 = rows.find((r) => r.label === "2024")!;
    expect(row2023.AAPL).toBe(100);
    expect(row2023.KO).toBe(200);
    expect(row2024.AAPL).toBe(110);
    expect(row2024.KO).toBe(220);
  });

  it("keeps the latest original date per layer+bucket when two points land in the same bucket", () => {
    const layers = [
      layer("AAPL", [
        ["2024-01-05", 1],
        ["2024-03-20", 2], // gleicher Bucket (Q1 2024), spaeteres Datum -> gewinnt
      ]),
    ];

    const rows = mergeLayers(layers, "quarter");

    expect(rows).toHaveLength(1);
    expect(rows[0].AAPL).toBe(2);
  });

  it("sorts rows chronologically by the latest original date, not alphabetically by bucket label", () => {
    // "Q4 2023" < "Q1 2024" alphabetisch ist FALSCH herum - die Sortierung
    // muss ueber das echte Datum laufen, nicht den Bucket-String.
    const layers = [
      layer("AAPL", [
        ["2024-01-15", 10], // Q1 2024
        ["2023-10-01", 20], // Q4 2023
      ]),
    ];

    const rows = mergeLayers(layers, "quarter");

    expect(rows.map((r) => r.label)).toEqual(["Q4 2023", "Q1 2024"]);
  });

  it("leaves a single-company series unmerged (date mode is a pass-through)", () => {
    const layers = [layer("AAPL", [["2024-09-30", 5]])];

    const rows = mergeLayers(layers, "date");

    expect(rows).toEqual([{ date: "2024-09-30", label: "2024-09-30", AAPL: 5 }]);
  });

  it("produces exactly one row per bucket even with three offset companies (chart-hover root-cause fix)", () => {
    const layers = [
      layer("AAPL", [["2024-09-30", 1]]),
      layer("MSFT", [["2024-06-30", 2]]),
      layer("KO", [["2024-12-31", 3]]),
    ];

    const rows = mergeLayers(layers, "year");

    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ label: "2024", AAPL: 1, MSFT: 2, KO: 3 });
  });
});

describe("extractTooltipRows (EVOLVING.md EV-031)", () => {
  it("returns one row per layer, even when the merged chart row has no value for it", () => {
    const layers = [layer("AAPL", []), layer("MSFT", []), layer("KO", [])];
    // Simuliert eine gemergte Chart-Zeile, in der nur AAPL und KO einen
    // Wert haben (MSFT fehlt in diesem Bucket) - Root Cause des
    // "nur 1-2 Firmen im Tooltip"-Bugs: der Recharts-Standard-Tooltip
    // haette MSFT hier stillschweigend weggelassen.
    const row = { date: "2024-12-31", label: "2024", AAPL: 100, KO: 300 };

    const rows = extractTooltipRows(row, layers);

    expect(rows).toHaveLength(3);
    expect(rows.find((r) => r.id === "AAPL")?.value).toBe(100);
    expect(rows.find((r) => r.id === "MSFT")?.value).toBeNull();
    expect(rows.find((r) => r.id === "KO")?.value).toBe(300);
  });

  it("returns an empty array when the row is undefined (nothing hovered / no data)", () => {
    const layers = [layer("AAPL", [])];

    expect(extractTooltipRows(undefined, layers)).toEqual([]);
  });

  it("preserves layer color and label for rendering", () => {
    const layers: ChartLayer[] = [{ id: "AAPL", label: "Apple Inc.", axis: "left", color: "#22d3ee", data: [] }];
    const row = { date: "2024-12-31", label: "2024", AAPL: 42 };

    const rows = extractTooltipRows(row, layers);

    expect(rows[0]).toEqual({ id: "AAPL", label: "Apple Inc.", color: "#22d3ee", value: 42 });
  });

  it("treats a non-numeric cell value as missing (defensive, should not normally occur)", () => {
    const layers = [layer("AAPL", [])];
    const row = { date: "2024-12-31", label: "2024", AAPL: "not-a-number" as unknown as number };

    const rows = extractTooltipRows(row, layers);

    expect(rows[0].value).toBeNull();
  });

  it("carries the layer's currency through to the tooltip row", () => {
    const layers: ChartLayer[] = [
      { id: "AAPL", label: "AAPL", axis: "left", color: "#000", data: [], currency: "USD" },
    ];
    const row = { date: "2024-12-31", label: "2024", AAPL: 42 };

    expect(extractTooltipRows(row, layers)[0].currency).toBe("USD");
  });
});

describe("layersCurrencyState (EVOLVING.md EV-023)", () => {
  function currencyLayer(id: string, currency?: string | null): ChartLayer {
    return { id, label: id, axis: "left", color: "#000", data: [], currency };
  }

  it("returns 'none' when no layer has a known currency (e.g. ratio/margin charts)", () => {
    const state = layersCurrencyState([currencyLayer("A", undefined), currencyLayer("B", null)]);

    expect(state).toEqual({ none: true });
  });

  it("returns 'uniform' when all layers share the same currency", () => {
    const state = layersCurrencyState([currencyLayer("A", "USD"), currencyLayer("B", "USD")]);

    expect(state).toEqual({ uniform: "USD" });
  });

  it("returns 'uniform' even if some layers have an unknown (null) currency, as long as the known ones agree", () => {
    const state = layersCurrencyState([currencyLayer("A", "USD"), currencyLayer("B", null)]);

    expect(state).toEqual({ uniform: "USD" });
  });

  it("returns 'mixed' with all distinct codes when companies report in different currencies", () => {
    const state = layersCurrencyState([currencyLayer("A", "USD"), currencyLayer("B", "EUR")]);

    expect(state).toEqual({ mixed: ["USD", "EUR"] });
  });

  it("treats an empty layer list as 'none'", () => {
    expect(layersCurrencyState([])).toEqual({ none: true });
  });
});

describe("filterSeriesByRange (EVOLVING.md EV-040)", () => {
  function series(dates: string[]): Array<{ date: string; value: number }> {
    return dates.map((date, i) => ({ date, value: i }));
  }

  it("returns the series unchanged for 'max'", () => {
    const s = series(["2020-01-01", "2024-01-01"]);
    expect(filterSeriesByRange(s, "max")).toEqual(s);
  });

  it("returns an empty series unchanged for any range", () => {
    expect(filterSeriesByRange([], "1y")).toEqual([]);
  });

  it("keeps only points within the last month, anchored on the series' own latest date", () => {
    const s = series(["2023-12-01", "2024-01-01", "2024-02-15", "2024-03-01"]);
    const result = filterSeriesByRange(s, "1m");
    // Anker = 2024-03-01 (neuester Punkt), Cutoff = 2024-02-01
    expect(result.map((p) => p.date)).toEqual(["2024-02-15", "2024-03-01"]);
  });

  it("accepts an explicit anchorDate override (EV-041: synchronized multi-company cut)", () => {
    const s = series(["2023-01-01", "2023-06-01", "2023-12-01"]);
    const result = filterSeriesByRange(s, "6m", "2024-01-01");
    // Cutoff = 2023-07-01 (6 Kalendermonate vor 2024-01-01) -> 2023-06-01 faellt raus
    expect(result.map((p) => p.date)).toEqual(["2023-12-01"]);
  });

  it("handles a single-point series (anchor = that point itself)", () => {
    const s = series(["2024-06-15"]);
    expect(filterSeriesByRange(s, "1m")).toEqual(s);
    expect(filterSeriesByRange(s, "5y")).toEqual(s);
  });

  it("handles unsorted input by still deriving the correct max-date anchor", () => {
    const s = series(["2024-01-01", "2020-01-01", "2024-06-01", "2022-01-01"]);
    const result = filterSeriesByRange(s, "1y");
    // Anker = 2024-06-01 (nicht der letzte Serien-Eintrag), Cutoff = 2023-06-01
    expect(result.map((p) => p.date).sort()).toEqual(["2024-01-01", "2024-06-01"]);
  });

  it("clamps a month-end anchor across a leap-year February (31.03.2024 - 1 month -> 29.02.2024)", () => {
    const s = series(["2024-02-20", "2024-03-31"]);
    const result = filterSeriesByRange(s, "1m");
    // Cutoff = 2024-02-29 (Schaltjahr) -> 2024-02-20 liegt davor und faellt raus
    expect(result.map((p) => p.date)).toEqual(["2024-03-31"]);
  });

  it("clamps a month-end anchor across a non-leap-year February (31.03.2023 - 1 month -> 28.02.2023)", () => {
    const s = series(["2023-02-27", "2023-03-31"]);
    const result = filterSeriesByRange(s, "1m");
    // Cutoff = 2023-02-28, also faellt 2023-02-27 heraus
    expect(result.map((p) => p.date)).toEqual(["2023-03-31"]);
  });

  it("covers every configured range option with a plausible cutoff", () => {
    const anchor = "2024-01-15";
    const s = series(["2018-06-01", "2019-06-01", "2021-06-01", "2022-06-01", "2023-06-01", "2023-08-01", "2023-11-01", "2023-12-20"]);
    const expectations: Record<Exclude<TimeRange, "max">, string[]> = {
      "1m": ["2023-12-20"],
      "2m": ["2023-12-20"],
      "3m": ["2023-11-01", "2023-12-20"],
      "6m": ["2023-08-01", "2023-11-01", "2023-12-20"],
      "1y": ["2023-06-01", "2023-08-01", "2023-11-01", "2023-12-20"],
      "2y": ["2022-06-01", "2023-06-01", "2023-08-01", "2023-11-01", "2023-12-20"],
      "5y": ["2019-06-01", "2021-06-01", "2022-06-01", "2023-06-01", "2023-08-01", "2023-11-01", "2023-12-20"],
    };

    for (const [range, expected] of Object.entries(expectations) as Array<[Exclude<TimeRange, "max">, string[]]>) {
      const result = filterSeriesByRange(s, range, anchor);
      expect(result.map((p) => p.date)).toEqual(expected);
    }
  });
});

describe("filterChartLayers (EVOLVING.md EV-041)", () => {
  it("returns layers unchanged for 'max'", () => {
    const layers = [layer("AAPL", [["2020-01-01", 1]])];
    expect(filterChartLayers(layers, "max")).toEqual(layers);
  });

  it("cuts every layer at a SHARED anchor (the latest date across all layers), not each layer's own latest point", () => {
    // AAPL's latest point (2023-09-30) is earlier than KO's (2023-12-31) -
    // without a shared anchor, AAPL would use its own (earlier) anchor and
    // keep a different window than KO, defeating the synchronized-cut point
    // of this function.
    const layers = [
      layer("AAPL", [
        ["2022-09-30", 1],
        ["2023-09-30", 2],
      ]),
      layer("KO", [
        ["2022-12-31", 3],
        ["2023-12-31", 4],
      ]),
    ];

    const result = filterChartLayers(layers, "1y");

    // Gemeinsamer Anker = 2023-12-31 (KOs neuester Punkt) -> Cutoff 2022-12-31
    expect(result.find((l) => l.id === "AAPL")!.data.map((p) => p.date)).toEqual(["2023-09-30"]);
    expect(result.find((l) => l.id === "KO")!.data.map((p) => p.date)).toEqual(["2022-12-31", "2023-12-31"]);
  });

  it("returns the layers unchanged when every layer's data is empty (no anchor derivable)", () => {
    const layers = [layer("AAPL", []), layer("KO", [])];
    expect(filterChartLayers(layers, "1y")).toEqual(layers);
  });

  it("preserves non-data fields (id, label, color, currency) on each returned layer", () => {
    const layers: ChartLayer[] = [
      { id: "AAPL", label: "Apple Inc.", axis: "left", color: "#22d3ee", currency: "USD", data: [{ date: "2024-01-01", value: 1 }] },
    ];

    const result = filterChartLayers(layers, "max");

    expect(result[0]).toMatchObject({ id: "AAPL", label: "Apple Inc.", axis: "left", color: "#22d3ee", currency: "USD" });
  });
});

describe("computePercentChange (EVOLVING.md EV-050)", () => {
  function series(values: number[]): Array<{ date: string; value: number }> {
    return values.map((value, i) => ({ date: `2020-0${i + 1}-01`, value }));
  }

  it("computes a positive percent change from first to last point", () => {
    expect(computePercentChange(series([100, 150]))).toEqual({ percent: 50 });
  });

  it("computes a negative percent change from first to last point", () => {
    expect(computePercentChange(series([200, 150]))).toEqual({ percent: -25 });
  });

  it("computes exactly 0% when start equals end", () => {
    expect(computePercentChange(series([100, 90, 100]))).toEqual({ percent: 0 });
  });

  it("ignores intermediate points - only first and last matter", () => {
    expect(computePercentChange(series([100, 999, -999, 200]))).toEqual({ percent: 100 });
  });

  it("returns 'zero-start' when the first point is 0 (division by zero)", () => {
    expect(computePercentChange(series([0, 50]))).toEqual({ percent: null, reason: "zero-start" });
  });

  it("returns 'negative-start' when the first point is negative (mathematically misleading ratio)", () => {
    expect(computePercentChange(series([-50, 100]))).toEqual({ percent: null, reason: "negative-start" });
  });

  it("returns 'insufficient' for a single-point series", () => {
    expect(computePercentChange(series([100]))).toEqual({ percent: null, reason: "insufficient" });
  });

  it("returns 'insufficient' for an empty series", () => {
    expect(computePercentChange([])).toEqual({ percent: null, reason: "insufficient" });
  });

  it("skips NaN/Infinity values at the edges inward to the next finite point", () => {
    const s = [
      { date: "2020-01-01", value: NaN },
      { date: "2020-02-01", value: 100 },
      { date: "2020-03-01", value: 150 },
      { date: "2020-04-01", value: Infinity },
    ];
    expect(computePercentChange(s)).toEqual({ percent: 50 });
  });

  it("returns 'insufficient' when fewer than two finite points remain after skipping NaN edges", () => {
    const s = [
      { date: "2020-01-01", value: NaN },
      { date: "2020-02-01", value: 100 },
      { date: "2020-03-01", value: NaN },
    ];
    expect(computePercentChange(s)).toEqual({ percent: null, reason: "insufficient" });
  });

  it("computes fractional percentages precisely (rounding is formatPercentChange's job, not this function's)", () => {
    const result = computePercentChange(series([300, 400]));
    expect(result).toEqual({ percent: (100 / 300) * 100 });
  });

  it("computes start/end by chronological date, not array position, for a descending (newest-first) series", () => {
    // Root-Cause-Regression (live in EV-051 beobachtet): manche Backend-
    // Serien (z. B. sec_source.py::sort_index(ascending=False)) liefern
    // Punkte neuestes Datum zuerst - ein naiver series[0]/series[last]-
    // Zugriff hätte hier Start (2006, 20) und Ende (2025, 409) vertauscht
    // und aus einem Wachstum eine falsche ~-95%-Badge gemacht.
    const s = [
      { date: "2025-12-31", value: 409 },
      { date: "2020-12-31", value: 275 },
      { date: "2015-12-31", value: 108 },
      { date: "2006-12-31", value: 20 },
    ];
    expect(computePercentChange(s)).toEqual({ percent: ((409 - 20) / 20) * 100 });
  });

  it("computes start/end by chronological date for arbitrarily unsorted (non-monotonic) input", () => {
    const s = [
      { date: "2022-01-01", value: 50 },
      { date: "2020-01-01", value: 100 },
      { date: "2024-01-01", value: 80 },
      { date: "2021-01-01", value: 30 },
    ];
    // frühestes Datum 2020 (100), spätestes 2024 (80)
    expect(computePercentChange(s)).toEqual({ percent: -20 });
  });
});

describe("formatPercentChange (EVOLVING.md EV-050/051)", () => {
  it("formats a positive result with a '+' sign, German comma, one decimal", () => {
    expect(formatPercentChange({ percent: 12.4 })).toBe("+12,4 %");
  });

  it("formats a negative result with the native '-' sign (no double sign)", () => {
    expect(formatPercentChange({ percent: -8.7 })).toBe("-8,7 %");
  });

  it("rounds 33.333...% to '+33,3 %'", () => {
    expect(formatPercentChange({ percent: 100 / 3 })).toBe("+33,3 %");
  });

  it("shows a neutral '±0,0 %' for an exact 0, not '+0,0 %'", () => {
    expect(formatPercentChange({ percent: 0 })).toBe("±0,0 %");
  });

  it("shows a neutral '±0,0 %' instead of a misleading '-0,0 %' when rounding produces negative zero", () => {
    expect(formatPercentChange({ percent: -0.04 })).toBe("±0,0 %");
  });

  it("shows 'n. v.' for every null reason (insufficient/zero-start/negative-start)", () => {
    expect(formatPercentChange({ percent: null, reason: "insufficient" })).toBe("n. v.");
    expect(formatPercentChange({ percent: null, reason: "zero-start" })).toBe("n. v.");
    expect(formatPercentChange({ percent: null, reason: "negative-start" })).toBe("n. v.");
  });
});

describe("isPercentChangeEligibleUnit (EVOLVING.md EV-051)", () => {
  it("is eligible for currency-unit metrics (revenue, market cap, cashflow, ...)", () => {
    expect(isPercentChangeEligibleUnit("currency")).toBe(true);
  });

  it("is not eligible for ratio metrics (EV/EBIT, P/B, ...)", () => {
    expect(isPercentChangeEligibleUnit("ratio")).toBe(false);
  });

  it("is not eligible for percent-unit metrics (margins, growth rates, ...)", () => {
    expect(isPercentChangeEligibleUnit("%")).toBe(false);
  });

  it("is not eligible when no unit is configured (undefined)", () => {
    expect(isPercentChangeEligibleUnit(undefined)).toBe(false);
  });
});

describe("normalizePriceSeries (EVOLVING.md EV-062)", () => {
  it("normalizes the base point itself to 0%", () => {
    const result = normalizePriceSeries([
      { date: "2024-01-01", value: 100 },
      { date: "2024-02-01", value: 150 },
    ]);
    expect(result[0]).toEqual({ date: "2024-01-01", value: 0 });
  });

  it("computes positive percent change relative to the base point", () => {
    const result = normalizePriceSeries([
      { date: "2024-01-01", value: 100 },
      { date: "2024-02-01", value: 150 },
    ]);
    expect(result[1]).toEqual({ date: "2024-02-01", value: 50 });
  });

  it("computes negative percent change relative to the base point", () => {
    const result = normalizePriceSeries([
      { date: "2024-01-01", value: 200 },
      { date: "2024-02-01", value: 150 },
    ]);
    expect(result[1]).toEqual({ date: "2024-02-01", value: -25 });
  });

  it("normalizes series of different lengths independently (each starts at its own 0%)", () => {
    const short = normalizePriceSeries([
      { date: "2024-01-01", value: 50 },
      { date: "2024-01-02", value: 55 },
    ]);
    const long = normalizePriceSeries([
      { date: "2023-06-01", value: 10 },
      { date: "2023-12-01", value: 20 },
      { date: "2024-01-01", value: 30 },
    ]);
    expect(short.map((p) => p.value)).toEqual([0, 10]);
    expect(long.map((p) => p.value)).toEqual([0, 100, 200]);
  });

  it("finds the base point by earliest date, not array position, for unsorted input", () => {
    const result = normalizePriceSeries([
      { date: "2024-03-01", value: 300 },
      { date: "2024-01-01", value: 100 },
      { date: "2024-02-01", value: 150 },
    ]);
    // frühestes Datum 2024-01-01 (100) ist die Basis, nicht der erste Array-Eintrag (300)
    expect(result.find((p) => p.date === "2024-01-01")!.value).toBe(0);
    expect(result.find((p) => p.date === "2024-03-01")!.value).toBe(200);
  });

  it("returns an empty series when the base point value is 0 (avoids division by zero)", () => {
    expect(normalizePriceSeries([{ date: "2024-01-01", value: 0 }])).toEqual([]);
  });

  it("returns an empty series for empty input", () => {
    expect(normalizePriceSeries([])).toEqual([]);
  });

  it("skips non-finite values but keeps the rest normalized", () => {
    const result = normalizePriceSeries([
      { date: "2024-01-01", value: 100 },
      { date: "2024-01-02", value: NaN },
      { date: "2024-01-03", value: 120 },
    ]);
    expect(result).toEqual([
      { date: "2024-01-01", value: 0 },
      { date: "2024-01-03", value: 20 },
    ]);
  });
});

describe("localIsoDate (EVOLVING.md CH-003)", () => {
  it("formats a local date as YYYY-MM-DD with zero padding", () => {
    expect(localIsoDate(new Date(2026, 0, 5))).toBe("2026-01-05");
  });

  it("uses the LOCAL calendar day, not UTC (no date jump after midnight in DE)", () => {
    // 00:30 lokale Zeit — toISOString() würde hier (UTC+1/+2) noch den
    // Vortag liefern; localIsoDate muss den lokalen Tag zurückgeben.
    const shortlyAfterMidnight = new Date(2026, 6, 19, 0, 30);
    expect(localIsoDate(shortlyAfterMidnight)).toBe("2026-07-19");
  });
});

describe("appendLivePoint (EVOLVING.md CH-003/CH-005)", () => {
  const series = [
    { date: "2026-07-16", value: 100 },
    { date: "2026-07-17", value: 102 },
  ];

  it("appends a new point when the latest row is older than today", () => {
    const result = appendLivePoint(series, 105, "2026-07-18");
    expect(result).toEqual([
      { date: "2026-07-16", value: 100 },
      { date: "2026-07-17", value: 102 },
      { date: "2026-07-18", value: 105 },
    ]);
  });

  it("replaces the latest value when its date equals today (intraday update, no duplicate date)", () => {
    const result = appendLivePoint(series, 105, "2026-07-17");
    expect(result).toEqual([
      { date: "2026-07-16", value: 100 },
      { date: "2026-07-17", value: 105 },
    ]);
  });

  it("finds the latest row by date, not array position", () => {
    const unsorted = [
      { date: "2026-07-17", value: 102 },
      { date: "2026-07-16", value: 100 },
    ];
    const result = appendLivePoint(unsorted, 105, "2026-07-17");
    expect(result).toEqual([
      { date: "2026-07-17", value: 105 },
      { date: "2026-07-16", value: 100 },
    ]);
  });

  it("returns the series unchanged for a null/undefined/non-finite live price", () => {
    expect(appendLivePoint(series, null, "2026-07-18")).toBe(series);
    expect(appendLivePoint(series, undefined, "2026-07-18")).toBe(series);
    expect(appendLivePoint(series, NaN, "2026-07-18")).toBe(series);
  });

  it("returns an empty series unchanged", () => {
    expect(appendLivePoint([], 105, "2026-07-18")).toEqual([]);
  });

  it("leaves the series unchanged when the latest row is (defensively) in the future", () => {
    expect(appendLivePoint(series, 105, "2026-07-16")).toBe(series);
  });

  it("does not mutate the input series", () => {
    const input = [{ date: "2026-07-16", value: 100 }];
    appendLivePoint(input, 105, "2026-07-17");
    expect(input).toEqual([{ date: "2026-07-16", value: 100 }]);
  });
});

describe("computeStartEndTicks (EVOLVING.md CH-003/CH-004)", () => {
  const fmt = (v: number) => v.toFixed(2);

  it("returns [start, end] picked by date, not array position", () => {
    const ticks = computeStartEndTicks(
      [
        { date: "2026-07-10", value: 80 },
        { date: "2026-06-19", value: 100 },
        { date: "2026-07-17", value: 130 },
      ],
      fmt
    );
    expect(ticks).toEqual([100, 130]);
  });

  it("returns [] for fewer than 2 valid points", () => {
    expect(computeStartEndTicks([], fmt)).toEqual([]);
    expect(computeStartEndTicks([{ date: "2026-07-17", value: 100 }], fmt)).toEqual([]);
    expect(
      computeStartEndTicks(
        [
          { date: "2026-07-16", value: NaN },
          { date: "2026-07-17", value: 100 },
        ],
        fmt
      )
    ).toEqual([]);
  });

  it("dedupes to [end] for a flat series (zero span)", () => {
    const ticks = computeStartEndTicks(
      [
        { date: "2026-07-16", value: 100 },
        { date: "2026-07-17", value: 100 },
      ],
      fmt
    );
    expect(ticks).toEqual([100]);
  });

  it("dedupes to [end] when start and end format identically", () => {
    const ticks = computeStartEndTicks(
      [
        { date: "2026-06-19", value: 100.001 },
        { date: "2026-07-01", value: 90 },
        { date: "2026-07-17", value: 100.004 },
      ],
      fmt
    );
    expect(ticks).toEqual([100.004]);
  });

  it("dedupes to [end] when start and end are vertically too close (< 25% of the span)", () => {
    // Spannweite 100 (50..150), |start-ende| = 10 -> 10% -> Überlappung.
    const ticks = computeStartEndTicks(
      [
        { date: "2026-06-19", value: 100 },
        { date: "2026-07-01", value: 50 },
        { date: "2026-07-10", value: 150 },
        { date: "2026-07-17", value: 110 },
      ],
      fmt
    );
    expect(ticks).toEqual([110]);
  });
});

describe("formatPriceTick (EVOLVING.md CH-003)", () => {
  it("prefixes USD values with $ and two decimals below 1000", () => {
    expect(formatPriceTick(56.789, "USD")).toBe("$56.79");
  });

  it("rounds values at or above 1000 without decimals (en-US grouping)", () => {
    expect(formatPriceTick(712345.6, "USD")).toBe("$712,346");
  });

  it("omits the $ prefix for non-USD and unknown currencies", () => {
    expect(formatPriceTick(56.789, "EUR")).toBe("56.79");
    expect(formatPriceTick(56.789, null)).toBe("56.79");
    expect(formatPriceTick(56.789, undefined)).toBe("56.79");
  });
});


describe("filterToLastDays (EVOLVING.md CH-007)", () => {
  it("keeps only points within the last N calendar days, anchored on the series' latest date", () => {
    const series = [
      { date: "2026-07-01", value: 10 },
      { date: "2026-07-10", value: 20 },
      { date: "2026-07-16", value: 30 },
      { date: "2026-07-17", value: 40 },
    ];

    const result = filterToLastDays(series, 7);

    expect(result).toEqual([
      { date: "2026-07-10", value: 20 },
      { date: "2026-07-16", value: 30 },
      { date: "2026-07-17", value: 40 },
    ]);
  });

  it("anchors on the latest date by value, not array position (unsorted input)", () => {
    const series = [
      { date: "2026-07-01", value: 10 },
      { date: "2026-07-17", value: 40 },
      { date: "2026-07-10", value: 20 },
    ];

    const result = filterToLastDays(series, 7);

    expect(result.map((p) => p.date).sort()).toEqual(["2026-07-10", "2026-07-17"]);
  });

  it("returns an empty series unchanged", () => {
    expect(filterToLastDays([], 7)).toEqual([]);
  });

  it("keeps every point when the window covers the whole series", () => {
    const series = [
      { date: "2026-07-16", value: 1 },
      { date: "2026-07-17", value: 2 },
    ];

    expect(filterToLastDays(series, 30)).toEqual(series);
  });
});
