import { INTL_LOCALES, type Locale } from "./config";

/** Ersetzt `{name}`-Platzhalter in einem Übersetzungs-String. Unbekannte
 * Platzhalter bleiben unverändert stehen (sichtbarer Fehler statt
 * stillschweigend verschluckt), fehlende params-Werte ebenso. */
export function interpolate(template: string, params?: Record<string, string | number>): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (match, key: string) =>
    Object.prototype.hasOwnProperty.call(params, key) ? String(params[key]) : match
  );
}

export type PluralForms = Partial<Record<Intl.LDMLPluralRule, string>> & { other: string };

/** Nativ über Intl.PluralRules statt einer Library-ICU-Engine — deckt auch
 * spätere Sprachen mit komplexeren Pluralregeln (z. B. Slawisch) korrekt ab,
 * ohne zusätzlichen Code (EVOLVING.md § 13). */
export function plural(locale: Locale, count: number, forms: PluralForms): string {
  const rule = new Intl.PluralRules(INTL_LOCALES[locale]).select(count);
  return forms[rule] ?? forms.other;
}
