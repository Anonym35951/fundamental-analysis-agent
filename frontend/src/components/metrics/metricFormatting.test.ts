import { describe, expect, it } from "vitest";
import { formatMonetary } from "./metricFormatting";

describe("formatMonetary (EVOLVING.md EV-022/EV-032)", () => {
  it("formats USD with a $ prefix", () => {
    expect(formatMonetary(1_200_000_000, "USD")).toBe("$1.2B");
  });

  it("formats EUR with a € prefix", () => {
    expect(formatMonetary(1_200_000_000, "EUR")).toBe("€1.2B");
  });

  it("formats GBP with a £ prefix", () => {
    expect(formatMonetary(1_200_000_000, "GBP")).toBe("£1.2B");
  });

  it("formats JPY with a ¥ prefix", () => {
    expect(formatMonetary(1_200_000_000, "JPY")).toBe("¥1.2B");
  });

  it("appends unknown ISO codes as a suffix instead of guessing a symbol", () => {
    expect(formatMonetary(1_200_000_000, "CNY")).toBe("1.2B CNY");
  });

  it("falls back to $ (current behavior) when no currency is given", () => {
    expect(formatMonetary(1_200_000_000, undefined)).toBe("$1.2B");
  });

  it("falls back to $ (current behavior) when currency is explicitly null", () => {
    expect(formatMonetary(1_200_000_000, null)).toBe("$1.2B");
  });

  it("falls back to $ when currency is an empty string", () => {
    expect(formatMonetary(42, "")).toBe("$42");
  });
});
