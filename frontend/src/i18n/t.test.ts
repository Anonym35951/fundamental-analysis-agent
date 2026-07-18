import { describe, expect, it } from "vitest";
import { interpolate, plural } from "./t";

describe("interpolate", () => {
  it("replaces a single placeholder", () => {
    expect(interpolate("Hallo {name}", { name: "Efe" })).toBe("Hallo Efe");
  });

  it("replaces multiple placeholders", () => {
    expect(interpolate("{a} von {b}", { a: 3, b: 10 })).toBe("3 von 10");
  });

  it("leaves the template unchanged without params", () => {
    expect(interpolate("Hallo {name}")).toBe("Hallo {name}");
  });

  it("leaves unknown placeholders untouched", () => {
    expect(interpolate("Hallo {name}", { other: "x" })).toBe("Hallo {name}");
  });

  it("passes plain templates through unchanged", () => {
    expect(interpolate("Kein Platzhalter hier.", { unused: "x" })).toBe("Kein Platzhalter hier.");
  });
});

describe("plural", () => {
  it("selects the 'one' form for count 1 in German", () => {
    expect(plural("de", 1, { one: "1 Kriterium", other: "{count} Kriterien" })).toBe("1 Kriterium");
  });

  it("selects the 'other' form for count 0 in German", () => {
    expect(plural("de", 0, { other: "{count} Kriterien" })).toBe("{count} Kriterien");
  });

  it("selects the 'other' form for count 5 in German", () => {
    expect(plural("de", 5, { one: "1 Kriterium", other: "{count} Kriterien" })).toBe("{count} Kriterien");
  });

  it("falls back to 'other' when no matching form is provided", () => {
    expect(plural("en", 1, { other: "{count} items" })).toBe("{count} items");
  });
});
