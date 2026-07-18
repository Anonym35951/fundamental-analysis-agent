/** EVOLVING.md § 13/14: generische Typ-Infrastruktur für das typisierte
 * Translation-Dictionary. `DotPaths<T>` liefert eine Literal-Union aller
 * gültigen Dot-Pfade eines Namespace-Baums — ein Tippfehler in einem
 * `t()`-Aufruf ist dadurch ein TS-Compile-Fehler, kein Laufzeit-Bug.
 * `MatchShape<T>` erzwingt, dass eine andere Sprache exakt dieselben Keys
 * hat wie das DE-Dictionary (fehlender/überzähliger Key = Compile-Fehler). */

export type TranslationNode = string | { readonly [key: string]: TranslationNode };

export type DotPaths<T> = T extends string
  ? never
  : {
      [K in keyof T & string]: T[K] extends string ? K : `${K}.${DotPaths<T[K]>}`;
    }[keyof T & string];

/** Erzwingt identische Schlüsselstruktur wie T, erlaubt aber andere
 * String-Werte (Übersetzungen) je Sprache. */
export type MatchShape<T> = T extends string ? string : { [K in keyof T]: MatchShape<T[K]> };

export function resolvePath(node: TranslationNode, path: string): string | undefined {
  let current: TranslationNode = node;
  for (const segment of path.split(".")) {
    if (typeof current === "string") return undefined;
    const next = current[segment];
    if (next === undefined) return undefined;
    current = next;
  }
  return typeof current === "string" ? current : undefined;
}
