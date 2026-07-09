import { useMemo, useState, type ReactNode } from "react";
import { TourStatusContext } from "./tourStatusContext";

/** Teilt mit, ob die globale mehrseitige Onboarding-Tour (AppTour) gerade
 * laeuft - genutzt von useFirstRunOnboarding, um nicht gleichzeitig auf
 * /app/analyze ein zweites Overlay zu zeigen. `currentStepData` spiegelt das
 * optionale `data`-Feld des aktiven Tour-Schritts (tourSteps.ts) nach aussen,
 * damit einzelne Seiten (z.B. AnalyzePage) ihren lokalen State passend zum
 * aktuellen Schritt umschalten koennen, ohne dass tourSteps.ts Seiten-State
 * kennen muss. */
export function TourStatusProvider({ children }: { children: ReactNode }) {
  const [isTourRunning, setIsTourRunning] = useState(false);
  const [currentStepData, setCurrentStepData] = useState<Record<string, unknown> | null>(null);

  const value = useMemo(
    () => ({ isTourRunning, setIsTourRunning, currentStepData, setCurrentStepData }),
    [isTourRunning, currentStepData]
  );

  return (
    <TourStatusContext.Provider value={value}>
      {children}
    </TourStatusContext.Provider>
  );
}
