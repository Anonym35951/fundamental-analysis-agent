/** DE-Dictionary — die Quelle der Wahrheit (EVOLVING.md § 13/14). Deutsche
 * Strings werden beim Migrieren jeder Phase VERBATIM aus dem bestehenden
 * Code hierher verschoben, nie umformuliert (Paritätsgebot § 2).
 *
 * Phase 1 (I18N-002): reines Namespace-Gerüst, noch kein String migriert —
 * jeder Namespace startet leer und wird in seiner jeweiligen Phase befüllt
 * (§ 22). Sobald ein Namespace Inhalt bekommt, zieht er in eine eigene Datei
 * um (`locales/de/<namespace>.ts`), analog zur EN-Seite. */

export const de = {
  common: {},
  nav: {},
  auth: {},
  landing: {},
  pricing: {},
  dashboard: {},
  analysis: {},
  customAnalysis: {},
  compare: {},
  account: {},
  billing: {},
  support: {},
  errors: {},
  validation: {},
  metrics: {},
  charts: {},
  tour: {},
  agent: {},
} as const;

export type Dictionary = typeof de;
export type Namespace = keyof Dictionary;
