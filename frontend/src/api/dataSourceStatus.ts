import { apiRequest } from "./client";

export type DataSourceStatusEntry = {
  name: string;
  status: "ok" | "down";
};

export async function getDataSourceStatus(): Promise<DataSourceStatusEntry[]> {
  const data = await apiRequest<{ sources: DataSourceStatusEntry[] }>("/status/data-sources");
  return data.sources;
}
