import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { EventData } from "react-joyride";
import { ACTIONS, EVENTS, STATUS } from "react-joyride";
import { getCurrentUser, completeOnboarding } from "../api/auth";
import { tourSteps } from "../components/onboarding/tourSteps";
import { useTourStatus } from "./useTourStatus";

const DASHBOARD_ROUTE = "/app/dashboard";
const RESTART_QUERY_PARAM = "startTour";

/** Steuert die seitenuebergreifende Onboarding-Tour (AppTour): entscheidet,
 * ob/wann sie startet, haelt den aktuellen Schritt und navigiert zwischen
 * den Ziel-Routen der einzelnen Stationen (tourSteps.ts). */
export function useAppTour(introDone: boolean) {
  const location = useLocation();
  const navigate = useNavigate();
  const { setIsTourRunning, setCurrentStepData } = useTourStatus();

  const [run, setRun] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [hasAutoChecked, setHasAutoChecked] = useState(false);

  // Expliziter Neustart ueber ?startTour=1 (Button auf der Account-Seite) -
  // funktioniert unabhaengig von onboarding_completed_at und unabhaengig
  // davon, ob der automatische Erst-Check in dieser Session schon gelaufen
  // ist. Eigener Effekt, damit er nicht durch hasAutoChecked blockiert wird.
  useEffect(() => {
    if (!introDone) return;

    const params = new URLSearchParams(location.search);
    if (params.get(RESTART_QUERY_PARAM) !== "1") return;

    // Reagiert auf einen URL-Query-Param (externes Signal) und loest
    // zusaetzlich eine Navigation als Seiteneffekt aus - kein waehrend des
    // Renders berechenbarer Wert, legitimer Effect-Fall
    // (LAUNCH_AUDIT.md P2-10).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setStepIndex(0);
    setRun(true);
    // Query-Param sofort entfernen, damit ein spaeterer Reload derselben URL
    // die Tour nicht ungewollt erneut startet.
    navigate(DASHBOARD_ROUTE, { replace: true });
  }, [introDone, location.search, navigate]);

  // Automatischer Erststart direkt nach Login/Registrierung auf dem
  // Dashboard (nicht mitten in einer Session auf einer beliebigen Seite).
  // Laeuft nur einmal pro Layout-Mount (d.h. einmal pro Login).
  //
  // hasAutoChecked wird bewusst erst NACH Abschluss von getCurrentUser()
  // gesetzt (im .then/.catch), nicht synchron davor: React StrictMode fuehrt
  // Effekte im Dev-Modus einmal als Mount->Cleanup->Remount-Zyklus aus. Wuerde
  // hasAutoChecked schon vor dem Fetch gesetzt, wuerde der erste (verworfene)
  // Durchlauf das Flag setzen und dann per Cleanup isCancelled=true setzen,
  // bevor sein Promise resolved - der zweite (echte) Durchlauf saehe
  // hasAutoChecked bereits true und wuerde nie selbst einen Fetch starten,
  // waehrend der urspruengliche Fetch durch isCancelled verworfen wird. Die
  // Tour wuerde dadurch nie automatisch starten.
  useEffect(() => {
    if (!introDone || hasAutoChecked || location.pathname !== DASHBOARD_ROUTE) {
      return;
    }

    let isCancelled = false;
    getCurrentUser()
      .then((user) => {
        if (isCancelled) return;
        setHasAutoChecked(true);
        if (!user.onboarding_completed_at) {
          setStepIndex(0);
          setRun(true);
        }
      })
      .catch(() => {
        // Lieber keine Tour als eine kaputte Anzeige - beim naechsten Mount
        // (z.B. Reload) wird es erneut versucht, da hasAutoChecked dann noch
        // false ist.
        if (isCancelled) return;
        setHasAutoChecked(true);
      });

    return () => {
      isCancelled = true;
    };
  }, [introDone, hasAutoChecked, location.pathname]);

  useEffect(() => {
    setIsTourRunning(run);
  }, [run, setIsTourRunning]);

  // Spiegelt das optionale `data`-Feld des aktiven Schritts nach aussen, so
  // dass Seiten wie AnalyzePage ihren lokalen Tab-State passend zum aktuell
  // gezeigten Tour-Schritt umschalten koennen (z.B. "Individuell"-Tab fuer
  // den "Eigene Analyse erstellen"-Schritt).
  useEffect(() => {
    if (!run) {
      setCurrentStepData(null);
      return;
    }
    const activeStep = tourSteps[stepIndex];
    setCurrentStepData((activeStep?.data as Record<string, unknown> | undefined) ?? null);
  }, [run, stepIndex, setCurrentStepData]);

  const finishTour = useCallback(() => {
    setRun(false);
    completeOnboarding().catch(() => {
      // Rein informativ fuers naechste Login - kein Nutzer-Feedback noetig,
      // wenn das Persistieren fehlschlaegt, startet die Tour hoechstens ein
      // weiteres Mal.
    });
  }, []);

  const handleEvent = useCallback(
    (data: EventData) => {
      const { status, action, index, type } = data;

      if (status === STATUS.FINISHED || status === STATUS.SKIPPED) {
        finishTour();
        return;
      }

      if (type === EVENTS.STEP_AFTER) {
        const nextIndex = action === ACTIONS.PREV ? index - 1 : index + 1;
        const nextStep = tourSteps[nextIndex];

        if (!nextStep) {
          finishTour();
          return;
        }

        if (nextStep.route !== location.pathname) {
          navigate(nextStep.route);
        }
        setStepIndex(nextIndex);
      }
    },
    [finishTour, navigate, location.pathname]
  );

  return { run, stepIndex, steps: tourSteps, handleEvent };
}
