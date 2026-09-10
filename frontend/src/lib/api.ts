export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`API ${path} failed: ${resp.status}`);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  health: () => getJson<import("./types").Health>("/health"),
  depots: () => getJson<import("./types").Depot[]>("/depots"),
  experiments: (params?: { fault_type?: string; depot_id?: number }) => {
    const qs = new URLSearchParams();
    if (params?.fault_type) qs.set("fault_type", params.fault_type);
    if (params?.depot_id !== undefined) qs.set("depot_id", String(params.depot_id));
    const suffix = qs.toString() ? `?${qs}` : "";
    return getJson<import("./types").Experiment[]>(`/experiments${suffix}`);
  },
  windowsForExperiment: (experimentId: number, limit = 5) =>
    getJson<import("./types").WindowSummary[]>(
      `/experiments/${experimentId}/windows?limit=${limit}`
    ),
  signalWindow: (windowId: number) =>
    getJson<import("./types").WindowDetail>(`/signal-window/${windowId}`),
  faultDistribution: () =>
    getJson<import("./types").FaultDistributionEntry[]>("/stats/fault-distribution"),
};
