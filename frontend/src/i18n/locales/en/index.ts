import type { MatchShape } from "../../types";
import type { Dictionary } from "../de";

/** EN-Dictionary — typisiert gegen das DE-Schema (`MatchShape<Dictionary>`):
 * ein fehlender oder überzähliger Namespace/Key ist ein TS-Compile-Fehler,
 * keine Übersetzungslücke, die erst zur Laufzeit auffällt (EVOLVING.md § 13,
 * § 15). Phase 1: leer wie das DE-Gerüst, wird synchron zu § 22 befüllt. */

export const en: MatchShape<Dictionary> = {
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
};
