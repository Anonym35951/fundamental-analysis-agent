import { useContext } from "react";
import { TourStatusContext } from "./tourStatusContext";

/** Liest den Status der globalen Onboarding-Tour (AppTour) - siehe
 * TourStatusContext.tsx fuer Provider und Kontext-Definition. */
export function useTourStatus() {
  const context = useContext(TourStatusContext);
  if (!context) {
    // Ausserhalb von AppLayout (z.B. auf oeffentlichen Seiten) laeuft nie
    // eine Tour - fester Default statt Pflicht-Provider ueberall.
    return {
      isTourRunning: false,
      setIsTourRunning: () => {},
      currentStepData: null,
      setCurrentStepData: () => {},
    };
  }
  return context;
}
