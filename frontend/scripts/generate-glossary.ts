/**
 * Generiert statische, vorgerenderte HTML-Seiten für das öffentliche
 * Kennzahlen-Glossar (SEO — die App ist sonst eine reine SPA ohne SSR, also
 * für Suchmaschinen praktisch unsichtbar). Läuft nach `vite build` und
 * schreibt zusätzliche Dateien direkt in den fertigen `dist`-Ordner.
 *
 * Bewusst kein React-SSR: Glossar-Seiten sind reine, nicht-interaktive
 * Inhaltsseiten (Definition + Formel je Kennzahl) auf demselben
 * Detailniveau wie die Info-Hover im Produkt — siehe
 * [[comanalysis-product-decisions]] (keine separate Methodik-Seite, bewusst
 * nicht tiefer als der Hover-Text). Ein schlankes, handgeschriebenes HTML-
 * Template ist hier robuster und schneller als eine volle SSG-Pipeline.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { METRICS_CONFIG } from "../src/config/metricsConfig";

// Muss für echte SEO-Metadaten (canonical, sitemap) auf die tatsächliche
// Produktions-Domain gesetzt werden, z. B. via `SITE_URL=https://... npm run build`.
const SITE_URL = process.env.SITE_URL ?? "https://example.com";
const DIST_DIR = join(import.meta.dirname, "..", "dist");
const GLOSSARY_DIR = join(DIST_DIR, "glossar");

type GlossaryEntry = {
  key: string;
  slug: string;
  label: string;
  description: string;
  formula?: string;
};

function slugify(key: string): string {
  return key.replaceAll("_", "-");
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pageShell(title: string, description: string, canonicalPath: string, bodyHtml: string): string {
  return `<!doctype html>
<html lang="de">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${escapeHtml(title)}</title>
    <meta name="description" content="${escapeHtml(description)}" />
    <link rel="canonical" href="${SITE_URL}${canonicalPath}" />
    <style>
      body { margin: 0; background: #f5f5f7; color: #18181b; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }
      .shell { max-width: 720px; margin: 0 auto; padding: 48px 24px 80px; }
      a { color: #18181b; }
      header a.brand { text-decoration: none; font-weight: 800; font-size: 1.2rem; }
      h1 { font-size: 2rem; letter-spacing: -0.02em; margin: 32px 0 16px; }
      .formula { background: #ffffff; border: 1px solid #e4e4e7; border-radius: 12px; padding: 16px 18px; font-family: ui-monospace, Menlo, monospace; font-size: 0.92rem; margin: 20px 0; }
      p { line-height: 1.7; font-size: 1.05rem; }
      .cta { display: inline-block; margin-top: 28px; padding: 12px 20px; border-radius: 999px; background: #18181b; color: #ffffff; text-decoration: none; font-weight: 700; }
      .back { display: inline-block; margin-top: 40px; font-size: 0.92rem; color: #52525b; text-decoration: none; }
      footer { margin-top: 60px; font-size: 0.85rem; color: #71717a; }
    </style>
  </head>
  <body>
    <div class="shell">
      <header><a class="brand" href="${SITE_URL}/">ComAnalysis</a></header>
      ${bodyHtml}
      <footer>&copy; ${new Date().getFullYear()} ComAnalysis</footer>
    </div>
  </body>
</html>
`;
}

function renderEntryPage(entry: GlossaryEntry): string {
  const formulaHtml = entry.formula
    ? `<div class="formula">${escapeHtml(entry.formula)}</div>`
    : "";

  const body = `
      <h1>${escapeHtml(entry.label)}</h1>
      <p>${escapeHtml(entry.description)}</p>
      ${formulaHtml}
      <a id="glossary-cta" class="cta" href="${SITE_URL}/register">Kostenlos analysieren</a>
      <br />
      <a class="back" href="${SITE_URL}/glossar">&larr; Zurück zum Glossar</a>
      <script>
        if (localStorage.getItem("access_token")) {
          var cta = document.getElementById("glossary-cta");
          cta.href = "${SITE_URL}/app/analyze";
          cta.textContent = "Zur Analyse";
        }
      </script>
  `;

  return pageShell(
    `${entry.label} – Definition & Formel | ComAnalysis`,
    entry.description,
    `/glossar/${entry.slug}`,
    body,
  );
}

function renderIndexPage(entries: GlossaryEntry[]): string {
  const items = entries
    .map((entry) => `<li><a href="${SITE_URL}/glossar/${entry.slug}">${escapeHtml(entry.label)}</a></li>`)
    .join("\n        ");

  const body = `
      <h1>Kennzahlen-Glossar</h1>
      <p>Definition und Berechnungsformel für alle Kennzahlen, die ComAnalysis aus SEC-Filings ableitet.</p>
      <ul>
        ${items}
      </ul>
  `;

  return pageShell(
    "Kennzahlen-Glossar | ComAnalysis",
    "Definition und Formel für alle Fundamentalkennzahlen von ComAnalysis, direkt aus SEC-Filings berechnet.",
    "/glossar",
    body,
  );
}

function buildSitemap(entries: GlossaryEntry[]): string {
  const urls = [
    `${SITE_URL}/`,
    `${SITE_URL}/glossar`,
    ...entries.map((entry) => `${SITE_URL}/glossar/${entry.slug}`),
  ];

  const urlEntries = urls.map((url) => `  <url><loc>${url}</loc></url>`).join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urlEntries}\n</urlset>\n`;
}

function main() {
  const entries: GlossaryEntry[] = Object.entries(METRICS_CONFIG)
    .filter(([, config]) => Boolean(config.description))
    .map(([key, config]) => ({
      key,
      slug: slugify(key),
      label: config.label,
      description: config.description!,
      formula: config.formula,
    }));

  mkdirSync(GLOSSARY_DIR, { recursive: true });

  for (const entry of entries) {
    const entryDir = join(GLOSSARY_DIR, entry.slug);
    mkdirSync(entryDir, { recursive: true });
    writeFileSync(join(entryDir, "index.html"), renderEntryPage(entry), "utf-8");
  }

  writeFileSync(join(GLOSSARY_DIR, "index.html"), renderIndexPage(entries), "utf-8");
  writeFileSync(join(DIST_DIR, "sitemap.xml"), buildSitemap(entries), "utf-8");

  console.log(`Glossar generiert: ${entries.length} Kennzahl-Seiten unter dist/glossar/`);
}

main();
