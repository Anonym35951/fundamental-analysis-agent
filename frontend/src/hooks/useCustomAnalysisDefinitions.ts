import { useEffect, useState } from "react";
import { useToast } from "../components/ui/useToast";
import { getCurrentUser } from "../api/auth";
import {
  createDefinition,
  deleteDefinition,
  getCustomMetricsCatalog,
  listDefinitions,
  updateDefinition,
} from "../api/customAnalysis";
import type { CustomAnalysisDefinition, MetricCatalogEntry, MetricSelection } from "../types/customAnalysis";

const FREE_PLAN_DEFINITION_LIMIT = 1;

/** Catalog + saved-definitions loading and CRUD, extracted from the old
 * standalone CustomAnalysisPage so AnalyzePage's "Individuell" tab can drive
 * the same save/rename/delete/free-plan-limit logic inline. */
export function useCustomAnalysisDefinitions() {
  const { showToast } = useToast();

  const [catalog, setCatalog] = useState<MetricCatalogEntry[]>([]);
  const [isLoadingCatalog, setIsLoadingCatalog] = useState(true);

  const [definitions, setDefinitions] = useState<CustomAnalysisDefinition[]>([]);
  const [isLoadingDefinitions, setIsLoadingDefinitions] = useState(true);

  const [userPlan, setUserPlan] = useState("free");
  const [monthlyRequestCount, setMonthlyRequestCount] = useState(0);
  const [monthlyRequestLimit, setMonthlyRequestLimit] = useState<number | null>(null);

  function refreshUsage() {
    return getCurrentUser()
      .then((user) => {
        setUserPlan(typeof user.plan === "string" ? user.plan : "free");
        setMonthlyRequestCount(user.monthly_request_count ?? 0);
        setMonthlyRequestLimit(user.monthly_request_limit ?? null);
      })
      .catch(() => setUserPlan("free"));
  }

  useEffect(() => {
    refreshUsage();
     
  }, []);

  useEffect(() => {
    getCustomMetricsCatalog()
      .then(setCatalog)
      .catch(() => showToast("Metrik-Katalog konnte nicht geladen werden.", "error"))
      .finally(() => setIsLoadingCatalog(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function reloadDefinitions() {
    setIsLoadingDefinitions(true);
    return listDefinitions()
      .then(setDefinitions)
      .catch(() => showToast("Gespeicherte Analysen konnten nicht geladen werden.", "error"))
      .finally(() => setIsLoadingDefinitions(false));
  }

  useEffect(() => {
    reloadDefinitions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function saveDefinition(name: string, metrics: MetricSelection[], editingId: number | null) {
    if (editingId) {
      await updateDefinition(editingId, { name, metrics });
      showToast("Analyse aktualisiert.", "success");
    } else {
      await createDefinition({ name, metrics });
      showToast("Analyse gespeichert.", "success");
    }
    await reloadDefinitions();
  }

  async function renameDefinition(id: number, name: string) {
    try {
      await updateDefinition(id, { name });
      await reloadDefinitions();
    } catch {
      showToast("Umbenennen fehlgeschlagen.", "error");
    }
  }

  async function removeDefinition(id: number) {
    try {
      await deleteDefinition(id);
      await reloadDefinitions();
      showToast("Analyse gelöscht.", "success");
    } catch {
      showToast("Löschen fehlgeschlagen.", "error");
    }
  }

  // Pre-emptive UI check so free-plan users see the upsell instead of a
  // failed request; the backend's check_can_save_definition remains the
  // actual source of truth.
  const canCreateNew = userPlan !== "free" || definitions.length < FREE_PLAN_DEFINITION_LIMIT;

  // Mirrors the backend's require_analysis_access check (api/core/dependencies.py)
  // so "Analyse starten" can be disabled before the request even goes out,
  // instead of only surfacing the 403 after the fact.
  const isAnalysisLimitReached =
    monthlyRequestLimit !== null && monthlyRequestCount >= monthlyRequestLimit;

  return {
    catalog,
    isLoadingCatalog,
    definitions,
    isLoadingDefinitions,
    canCreateNew,
    isAnalysisLimitReached,
    refreshUsage,
    saveDefinition,
    renameDefinition,
    removeDefinition,
    reloadDefinitions,
  };
}
