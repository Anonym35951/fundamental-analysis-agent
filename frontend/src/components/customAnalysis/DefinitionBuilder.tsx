import { useState } from "react";
import { Button, Card, Input, theme } from "../ui";
import type { MetricCatalogEntry, MetricSelection } from "../../types/customAnalysis";
import MetricCatalogPicker from "./MetricCatalogPicker";

type DefinitionBuilderProps = {
  catalog: MetricCatalogEntry[];
  isLoadingCatalog: boolean;
  initialName?: string;
  initialMetrics?: MetricSelection[];
  isSaving: boolean;
  onSave: (name: string, metrics: MetricSelection[]) => void;
  onCancel: () => void;
};

/** Name field + metric picker for creating/editing a saved definition. The
 * ad-hoc (run-once, no save) flow lives in AdHocAnalysisPanel instead, which
 * shares the same MetricCatalogPicker but needs a symbol field and an
 * immediate "Analyse starten" action rather than a name + save button. */
export default function DefinitionBuilder({
  catalog,
  isLoadingCatalog,
  initialName = "",
  initialMetrics = [],
  isSaving,
  onSave,
  onCancel,
}: DefinitionBuilderProps) {
  const [name, setName] = useState(initialName);
  const [metrics, setMetrics] = useState<MetricSelection[]>(initialMetrics);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
      <Card variant="glass">
        <label style={{ display: "block", marginBottom: "8px", color: theme.colors.textSecondary, fontWeight: 600 }}>
          Name der Analyse
        </label>
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="z. B. Meine Qualitäts-Checkliste" />
      </Card>

      <MetricCatalogPicker
        catalog={catalog}
        isLoadingCatalog={isLoadingCatalog}
        initialMetrics={initialMetrics}
        onChange={setMetrics}
      />

      <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
        <Button variant="ghost" onClick={onCancel} style={{ flexShrink: 0 }}>
          Abbrechen
        </Button>
        <Button
          disabled={isSaving || !name.trim() || metrics.length === 0}
          onClick={() => onSave(name.trim(), metrics)}
          style={{ flexShrink: 0 }}
        >
          {isSaving ? "Wird gespeichert..." : "Speichern"}
        </Button>
      </div>
    </div>
  );
}
