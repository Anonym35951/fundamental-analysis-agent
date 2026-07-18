import { afterEach, describe, expect, it, vi } from "vitest";
import { detectInitialLocale, persistLocale } from "./detect";
import { LOCALE_STORAGE_KEY } from "./config";

/** Kein jsdom im Projekt (bewusst — vitest läuft standardmäßig in "node",
 * siehe frontend/src/components/charts/chartUtils.test.ts). localStorage/
 * navigator werden hier als einfache Objekte gestubbt, keine echte DOM-Umgebung
 * nötig, weil detect.ts nur diese beiden Globals liest/schreibt. */

function stubEnv(opts: { stored?: string | null; languages?: string[] }) {
  const store = new Map<string, string>();
  if (opts.stored) store.set(LOCALE_STORAGE_KEY, opts.stored);

  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
  });
  vi.stubGlobal("navigator", { languages: opts.languages ?? [], language: opts.languages?.[0] ?? "" });

  return store;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("detectInitialLocale", () => {
  it("prefers a valid stored locale over the browser language", () => {
    stubEnv({ stored: "en", languages: ["de-DE"] });
    expect(detectInitialLocale()).toBe("en");
  });

  it("ignores an invalid stored value and falls back to the browser language", () => {
    stubEnv({ stored: "fr", languages: ["en-US", "de-DE"] });
    expect(detectInitialLocale()).toBe("en");
  });

  it("matches a supported locale via the language prefix (en-GB -> en)", () => {
    stubEnv({ languages: ["en-GB"] });
    expect(detectInitialLocale()).toBe("en");
  });

  it("falls back to German when nothing matches", () => {
    stubEnv({ languages: ["fr-FR"] });
    expect(detectInitialLocale()).toBe("de");
  });

  it("falls back to German with no stored value and no browser languages", () => {
    stubEnv({});
    expect(detectInitialLocale()).toBe("de");
  });
});

describe("persistLocale", () => {
  it("writes the locale under the shared storage key", () => {
    const store = stubEnv({});
    persistLocale("en");
    expect(store.get(LOCALE_STORAGE_KEY)).toBe("en");
  });
});
