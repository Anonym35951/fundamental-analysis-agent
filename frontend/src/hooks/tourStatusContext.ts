import { createContext } from "react";

export type TourStatusContextValue = {
  isTourRunning: boolean;
  setIsTourRunning: (running: boolean) => void;
  currentStepData: Record<string, unknown> | null;
  setCurrentStepData: (data: Record<string, unknown> | null) => void;
};

export const TourStatusContext = createContext<TourStatusContextValue | null>(null);
