import { Joyride } from "react-joyride";
import { useAppTour } from "../../hooks/useAppTour";
import { theme } from "../ui/theme";

type AppTourProps = {
  /** Die Tour darf erst starten, nachdem IntroOverlay fertig ist - sonst
   * ueberlappen sich beide Overlays direkt nach dem Login. */
  introDone: boolean;
};

/** Gefuehrte, seitenuebergreifende Tour fuer Erstnutzer (Dashboard -> Analyse
 * -> Vergleich -> Account -> Billing). Rendert einmalig in AppLayout, nicht
 * pro Unterseite. */
function AppTour({ introDone }: AppTourProps) {
  const { run, stepIndex, steps, handleEvent } = useAppTour(introDone);

  return (
    <Joyride
      run={run}
      stepIndex={stepIndex}
      steps={steps}
      continuous
      onEvent={handleEvent}
      options={{
        primaryColor: "var(--color-chrome-strong)",
        backgroundColor: "var(--color-panel)",
        textColor: "var(--color-text-primary)",
        arrowColor: "var(--color-panel)",
        overlayColor: "rgba(0, 0, 0, 0.6)",
        showProgress: true,
        skipBeacon: true,
        buttons: ["back", "close", "primary", "skip"],
        closeButtonAction: "skip",
        targetWaitTimeout: 3000,
        zIndex: 9000,
      }}
      styles={{
        tooltip: {
          borderRadius: theme.radius.md,
          border: `1px solid ${theme.glass.elevated.border}`,
        },
        tooltipContent: {
          padding: "8px 4px",
          textAlign: "left",
        },
        tooltipTitle: {
          fontSize: "1.1rem",
          fontWeight: 700,
        },
        buttonPrimary: {
          borderRadius: theme.radius.pill,
          fontWeight: 700,
        },
        buttonSkip: {
          color: theme.colors.textMuted,
        },
      }}
      locale={{
        back: "Zurück",
        close: "Schließen",
        last: "Fertig",
        next: "Weiter",
        nextWithProgress: "Weiter ({current} von {total})",
        skip: "Tour überspringen",
      }}
    />
  );
}

export default AppTour;
