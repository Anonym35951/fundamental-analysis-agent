import { describe, expect, it } from "vitest";
import { de } from "./locales/de";
import { en } from "./locales/en";
import type { TranslationNode } from "./types";

/** EVOLVING.md § 15/27: Gürtel zusätzlich zum TS-Compile-Zeit-Hosenträger
 * (MatchShape<Dictionary> in locales/en/index.ts) — fängt zusätzlich leere
 * Strings, die TS nicht sieht. */

function collectEntries(node: TranslationNode, prefix = ""): Array<[string, string]> {
  if (typeof node === "string") return [[prefix, node]];
  return Object.entries(node).flatMap(([key, value]) => collectEntries(value, prefix ? `${prefix}.${key}` : key));
}

describe("i18n completeness", () => {
  it("DE and EN expose the exact same set of translation keys", () => {
    const deKeys = collectEntries(de).map(([key]) => key).sort();
    const enKeys = collectEntries(en).map(([key]) => key).sort();
    expect(enKeys).toEqual(deKeys);
  });

  it("no DE translation value is an empty string", () => {
    const empty = collectEntries(de).filter(([, value]) => value.trim() === "");
    expect(empty).toEqual([]);
  });

  it("no EN translation value is an empty string", () => {
    const empty = collectEntries(en).filter(([, value]) => value.trim() === "");
    expect(empty).toEqual([]);
  });
});
