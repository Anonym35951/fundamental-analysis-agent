import { describe, expect, it } from "vitest";
import { formatCompactNumber as formatCompactNumberChart, formatPercentChange } from "../components/charts/chartUtils";
import { formatCompactNumber as formatCompactNumberMetric, formatMetricValue } from "../components/metrics/metricFormatting";

/** EVOLVING.md § Internationalisierung, I18N-001: friert die heutigen
 * deutschen Formatter-Ausgaben als Literal-Erwartungswerte ein, BEVOR
 * format.ts (I18N-003) sie delegiert. Diese Datei darf durch die
 * i18n-Umstellung niemals rot werden — jede Abweichung ist ein
 * Paritätsbruch, kein zu behebender Bug in diesem Test. */

describe("chartUtils.formatCompactNumber — DE characterization", () => {
  it.each([
    [0, "0"],
    [1, "1"],
    [-1, "-1"],
    [999, "999"],
    [1000, "1.0 Tsd."],
    [-1000, "-1.0 Tsd."],
    [1500, "1.5 Tsd."],
    [9999, "10.0 Tsd."],
    [10000, "10 Tsd."],
    [999999, "1000 Tsd."],
    [1000000, "1.0 Mio."],
    [-1000000, "-1.0 Mio."],
    [1500000, "1.5 Mio."],
    [9999999, "10.0 Mio."],
    [10000000, "10 Mio."],
    [999999999, "1000 Mio."],
    [1000000000, "1.0 Mrd."],
    [-1000000000, "-1.0 Mrd."],
    [1500000000, "1.5 Mrd."],
    [9999999999, "10.0 Mrd."],
    [10000000000, "10 Mrd."],
    [123456789.5, "123 Mio."],
    [0.5, "0,5"],
    [-0.5, "-0,5"],
  ])("formats %s as %s", (value, expected) => {
    expect(formatCompactNumberChart(value)).toBe(expected);
  });
});

describe("chartUtils.formatPercentChange — DE characterization", () => {
  it.each([
    [{ percent: null, reason: "insufficient" as const }, "n. v."],
    [{ percent: 0 }, "±0,0 %"],
    [{ percent: -0.001 }, "±0,0 %"],
    [{ percent: 0.04 }, "±0,0 %"],
    [{ percent: 14.2 }, "+14,2 %"],
    [{ percent: -14.2 }, "-14,2 %"],
    [{ percent: 100 }, "+100,0 %"],
    [{ percent: -100 }, "-100,0 %"],
    [{ percent: 0.05 }, "+0,1 %"],
    [{ percent: -0.05 }, "±0,0 %"],
    [{ percent: 999.99 }, "+1.000,0 %"],
  ])("formats %j as %s", (result, expected) => {
    expect(formatPercentChange(result)).toBe(expected);
  });
});

describe("metricFormatting.formatCompactNumber — characterization (K/M/B, bewusst nicht de-DE)", () => {
  it.each([
    [0, "0"],
    [1, "1"],
    [-1, "-1"],
    [999, "999"],
    [1000, "1K"],
    [-1000, "-1K"],
    [1500, "1.5K"],
    [999999, "1000K"],
    [1000000, "1M"],
    [-1000000, "-1M"],
    [1500000, "1.5M"],
    [999999999, "1000M"],
    [1000000000, "1B"],
    [-1000000000, "-1B"],
    [1500000000, "1.5B"],
    [123456789.5, "123.46M"],
    [0.5, "0.5"],
    [-0.5, "-0.5"],
    [1234.5, "1.23K"],
    [10000, "10K"],
    [100000, "100K"],
  ])("formats %s as %s", (value, expected) => {
    expect(formatCompactNumberMetric(value)).toBe(expected);
  });
});

describe("metricFormatting.formatMetricValue — booleans and fallbacks", () => {
  it("formats true as Ja", () => {
    expect(formatMetricValue(true)).toBe("Ja");
  });

  it("formats false as Nein", () => {
    expect(formatMetricValue(false)).toBe("Nein");
  });

  it("formats null as em dash", () => {
    expect(formatMetricValue(null)).toBe("—");
  });

  it("formats undefined as em dash", () => {
    expect(formatMetricValue(undefined)).toBe("—");
  });
});
