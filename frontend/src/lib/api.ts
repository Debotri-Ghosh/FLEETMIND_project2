export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!resp.ok) {
    const detail = await extractDetail(resp);
    throw new ApiError(detail ?? `API ${path} failed: ${resp.status}`, resp.status);
  }
  return resp.json() as Promise<T>;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** FastAPI's HTTPException body is `{"detail": "..."}`; surface that message
 * when present so callers (e.g. a failed form submit) can show the real
 * reason -- "equipment 7 not found" -- instead of a bare status code. */
async function extractDetail(resp: Response): Promise<string | undefined> {
  try {
    const data = await resp.clone().json();
    return typeof data?.detail === "string" ? data.detail : undefined;
  } catch {
    return undefined;
  }
}

async function sendJson<T>(
  method: "POST" | "PATCH",
  path: string,
  body: unknown
): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const detail = await extractDetail(resp);
    throw new ApiError(detail ?? `API ${method} ${path} failed: ${resp.status}`, resp.status);
  }
  return resp.json() as Promise<T>;
}

function toQueryString(params: Record<string, string | number | boolean | undefined>) {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) qs.set(key, String(value));
  }
  const s = qs.toString();
  return s ? `?${s}` : "";
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
  modelPerformance: () =>
    getJson<import("./types").ModelPerformance>("/stats/model-performance"),
  productSummary: () =>
    getJson<import("./types").ProductSummary>("/stats/product-summary"),
  replayStatus: () => getJson<import("./types").ReplayStatus>("/replay/status"),
  replayStart: (params?: { interval_seconds?: number; batch_size?: number }) =>
    sendJson<import("./types").ReplayStatus>("POST", "/replay/start", params ?? {}),
  replayStop: () =>
    sendJson<import("./types").ReplayStatus>("POST", "/replay/stop", {}),
  replayStep: () =>
    sendJson<import("./types").ReplayStepResult>("POST", "/replay/step", {}),
  replayReset: () =>
    sendJson<Record<string, number>>("POST", "/replay/reset", {}),
  customers: () => getJson<import("./types").Customer[]>("/customers"),
  technicians: (params?: { status?: string }) =>
    getJson<import("./types").Technician[]>(
      `/technicians${toQueryString({ status: params?.status })}`
    ),
  equipment: (params?: {
    depot_id?: number;
    customer_id?: number;
    status?: string;
  }) =>
    getJson<import("./types").Equipment[]>(
      `/equipment${toQueryString({
        depot_id: params?.depot_id,
        customer_id: params?.customer_id,
        status: params?.status,
      })}`
    ),
  serviceRequests: (params?: {
    customer_id?: number;
    depot_id?: number;
    equipment_id?: number;
    status?: string;
    ai_flagged?: boolean;
  }) =>
    getJson<import("./types").ServiceRequest[]>(
      `/service-requests${toQueryString({
        customer_id: params?.customer_id,
        depot_id: params?.depot_id,
        equipment_id: params?.equipment_id,
        status: params?.status,
        ai_flagged: params?.ai_flagged,
      })}`
    ),
  createServiceRequest: (payload: import("./types").ServiceRequestCreate) =>
    sendJson<import("./types").ServiceRequest>("POST", "/service-requests", payload),
  workOrders: (params?: {
    technician_id?: number;
    status?: string;
    service_request_id?: number;
  }) =>
    getJson<import("./types").WorkOrder[]>(
      `/work-orders${toQueryString({
        technician_id: params?.technician_id,
        status: params?.status,
        service_request_id: params?.service_request_id,
      })}`
    ),
  createWorkOrder: (payload: { service_request_id: number; technician_id: number }) =>
    sendJson<import("./types").WorkOrder>("POST", "/work-orders", payload),
  updateWorkOrderStatus: (
    workOrderId: number,
    status: import("./types").WorkOrderStatus
  ) =>
    sendJson<import("./types").WorkOrder>(
      "PATCH",
      `/work-orders/${workOrderId}/status`,
      { status }
    ),
};
