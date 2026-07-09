import { useEffect, useState } from "react";
import { getAnalysisHistory } from "../api/analysis";
import { useTourStatus } from "./useTourStatus";

const DISMISSED_KEY = "onboarding_analyze_dismissed";

/** Drives the 3-Schritte-First-Run auf der Analyse-Seite (Beispiel-Ticker
 * vorschlagen -> Analyse ausfuehren -> Hinweis auf den Quellen-Badge): der
 * Erstnutzer soll das Transparenz-Differenzierungsmerkmal (Quellenangabe pro
 * Kennzahl) in den ersten Minuten erleben statt es zu uebersehen.
 *
 * "Erstnutzer" heisst hier: noch keine einzige Analyse in der Historie und
 * der Hinweis wurde in diesem Browser noch nicht weggeklickt. Ein Fehler beim
 * Laden der Historie unterdrueckt den Hinweis (lieber kein Onboarding als
 * eine kaputte Anzeige fuer wiederkehrende Nutzer). */
export function useFirstRunOnboarding() {
  const [isFirstRun, setIsFirstRun] = useState(false);
  const { isTourRunning } = useTourStatus();

  useEffect(() => {
    if (localStorage.getItem(DISMISSED_KEY) === "1") return;
    // Solange die globale, seitenuebergreifende Tour (AppTour) laeuft, nicht
    // zusaetzlich dieses In-Page-Overlay zeigen - sonst ueberlappen sich
    // zwei Hinweise auf derselben Seite.
    if (isTourRunning) return;

    let isCancelled = false;
    getAnalysisHistory()
      .then((history) => {
        if (!isCancelled && history.length === 0) setIsFirstRun(true);
      })
      .catch(() => {
        // Lieber kein Onboarding als eine kaputte Anzeige.
      });

    return () => {
      isCancelled = true;
    };
  }, [isTourRunning]);

  function dismissOnboarding() {
    localStorage.setItem(DISMISSED_KEY, "1");
    setIsFirstRun(false);
  }

  return { isFirstRun, dismissOnboarding };
}
