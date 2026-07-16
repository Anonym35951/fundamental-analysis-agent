/** EV-134: einzige Quelle der Wahrheit für den Frequenz-Typ im Frontend -
 * vorher als `CustomAnalysisFrequency`/`CompareFrequency` an zwei Stellen
 * separat als `"annual" | "quarterly"` dupliziert. "ttm" delegiert backend-
 * seitig 1:1 auf den Annual-Pfad (agent/frequency.py resolve_ttm_alias). */
export type Frequency = "annual" | "quarterly" | "ttm";
