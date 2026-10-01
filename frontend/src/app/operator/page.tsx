"use client";

import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { Fragment, useState } from "react";
import { api } from "@/lib/api";
import { RequireAuth, useAuth } from "../auth-provider";
import {
  ErrorRetry,
  EvidencePanel,
  Nav,
  PriorityBadge,
  ReplayControls,
  StatusBadge,
} from "../components";
import type { WorkOrderStatus } from "@/lib/types";

const NEXT_STATUS: Partial<Record<WorkOrderStatus, WorkOrderStatus>> = {
  assigned: "traveling",
  traveling: "in_progress",
  in_progress: "completed",
};

function EquipmentBoard({ depotId }: { depotId: number | null }) {
  const depots = useQuery({ queryKey: ["depots"], queryFn: api.depots });
  const equipment = useQuery({
    queryKey: ["equipment", "board", depotId],
    queryFn: () => api.equipment({ depot_id: depotId ?? undefined }),
  });

  if (equipment.isLoading) return <p className="text-slate-500">Loading equipment…</p>;
  if (equipment.isError)
    return <ErrorRetry message="Failed to load equipment." onRetry={() => equipment.refetch()} />;
  if (!equipment.data?.length)
    return <p className="text-slate-500">No equipment found.</p>;

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left text-slate-500">
          <th className="py-2">Equipment</th>
          <th className="py-2">Site</th>
          <th className="py-2">Type</th>
          <th className="py-2">Status</th>
        </tr>
      </thead>
      <tbody>
        {equipment.data.map((eq) => (
          <tr key={eq.id} className="border-b last:border-0">
            <td className="py-2 font-medium">{eq.name}</td>
            <td className="py-2">
              {depots.data?.find((d) => d.id === eq.depot_id)?.name ?? eq.depot_id}
            </td>
            <td className="py-2">{eq.equipment_type}</td>
            <td className="py-2">
              <StatusBadge status={eq.status} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function AiFlaggedPanel() {
  const requests = useQuery({
    queryKey: ["service-requests", "ai-flagged"],
    queryFn: () => api.serviceRequests({ ai_flagged: true }),
  });
  const equipment = useQuery({ queryKey: ["equipment", "all"], queryFn: () => api.equipment() });
  const [openEvidenceFor, setOpenEvidenceFor] = useState<number | null>(null);

  if (requests.isLoading) return <p className="text-slate-500">Loading AI-flagged requests…</p>;
  if (requests.isError)
    return (
      <ErrorRetry
        message="Failed to load AI-flagged requests."
        onRetry={() => requests.refetch()}
      />
    );
  if (!requests.data?.length)
    return (
      <p className="text-slate-500">
        No AI-flagged requests — the agent pipeline hasn&apos;t raised any HIGH/CRITICAL risk
        decisions that don&apos;t already have an open request.
      </p>
    );

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left text-slate-500">
          <th className="py-2">Equipment</th>
          <th className="py-2">Predicted Fault</th>
          <th className="py-2">Risk Score</th>
          <th className="py-2">Priority</th>
          <th className="py-2">Status</th>
          <th className="py-2">Agent Decision</th>
          <th className="py-2">Evidence</th>
        </tr>
      </thead>
      <tbody>
        {requests.data.map((r) => (
          <Fragment key={r.id}>
            <tr className="border-b align-top last:border-0">
              <td className="py-2 font-medium">
                {equipment.data?.find((e) => e.id === r.equipment_id)?.name ??
                  `Equipment #${r.equipment_id}`}
              </td>
              <td className="py-2">{r.predicted_fault ?? "—"}</td>
              <td className="py-2">
                {r.agent_risk_score !== null ? `${(r.agent_risk_score * 100).toFixed(0)}%` : "—"}
              </td>
              <td className="py-2">
                <PriorityBadge priority={r.priority} />
              </td>
              <td className="py-2">
                <StatusBadge status={r.status} />
              </td>
              <td className="max-w-md py-2 text-slate-600">{r.agent_decision_text ?? "—"}</td>
              <td className="py-2">
                <button
                  type="button"
                  onClick={() => setOpenEvidenceFor(openEvidenceFor === r.id ? null : r.id)}
                  className="whitespace-nowrap rounded border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                >
                  {openEvidenceFor === r.id ? "Hide evidence" : "Show evidence"}
                </button>
              </td>
            </tr>
            {openEvidenceFor === r.id && (
              <tr className="border-b last:border-0">
                <td colSpan={7} className="py-3">
                  <EvidencePanel requestId={r.id} />
                </td>
              </tr>
            )}
          </Fragment>
        ))}
      </tbody>
    </table>
  );
}

function DispatchQueue() {
  const queryClient = useQueryClient();
  const requests = useQuery({
    queryKey: ["service-requests", "all"],
    queryFn: () => api.serviceRequests(),
  });
  const technicians = useQuery({ queryKey: ["technicians"], queryFn: () => api.technicians() });
  const [selectedTechnician, setSelectedTechnician] = useState<Record<number, number>>({});

  const dispatch = useMutation({
    mutationFn: (vars: { serviceRequestId: number; technicianId: number }) =>
      api.createWorkOrder({
        service_request_id: vars.serviceRequestId,
        technician_id: vars.technicianId,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["service-requests"] });
      queryClient.invalidateQueries({ queryKey: ["work-orders"] });
    },
  });

  const pending = requests.data?.filter(
    (r) => r.status === "reported" || r.status === "diagnosed"
  );

  if (requests.isLoading) return <p className="text-slate-500">Loading requests…</p>;
  if (requests.isError)
    return <ErrorRetry message="Failed to load requests." onRetry={() => requests.refetch()} />;
  if (!pending?.length)
    return <p className="text-slate-500">Nothing awaiting dispatch.</p>;

  const noTechnicians = technicians.isSuccess && technicians.data.length === 0;

  return (
    <div className="space-y-3">
      {technicians.isError && (
        <ErrorRetry
          message="Failed to load technicians -- dispatch is unavailable until this loads."
          onRetry={() => technicians.refetch()}
        />
      )}
      {noTechnicians && (
        <p className="text-sm text-amber-700">
          No technicians available. Dispatch is disabled until a technician is added.
        </p>
      )}
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-slate-500">
            <th className="py-2">Request</th>
            <th className="py-2">Priority</th>
            <th className="py-2">Status</th>
            <th className="py-2">Technician</th>
            <th className="py-2"></th>
          </tr>
        </thead>
        <tbody>
          {pending.map((r) => (
            <tr key={r.id} className="border-b last:border-0">
              <td className="py-2 font-medium">{r.title}</td>
              <td className="py-2">
                <PriorityBadge priority={r.priority} />
              </td>
              <td className="py-2">
                <StatusBadge status={r.status} />
              </td>
              <td className="py-2">
                <select
                  className="rounded border bg-white px-2 py-1 disabled:opacity-50"
                  disabled={noTechnicians}
                  value={selectedTechnician[r.id] ?? ""}
                  onChange={(e) =>
                    setSelectedTechnician((prev) => ({
                      ...prev,
                      [r.id]: Number(e.target.value),
                    }))
                  }
                >
                  <option value="" disabled>
                    {noTechnicians ? "No technicians available" : "Select technician…"}
                  </option>
                  {technicians.data?.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name} ({t.status})
                    </option>
                  ))}
                </select>
              </td>
              <td className="py-2">
                <button
                  disabled={!selectedTechnician[r.id] || dispatch.isPending || noTechnicians}
                  onClick={() =>
                    dispatch.mutate({
                      serviceRequestId: r.id,
                      technicianId: selectedTechnician[r.id],
                    })
                  }
                  className="rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white disabled:opacity-50"
                >
                  Dispatch
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {dispatch.isError && (
        <p className="text-sm text-red-600">
          {dispatch.error instanceof Error ? dispatch.error.message : "Failed to dispatch."}
        </p>
      )}
    </div>
  );
}

function WorkOrderQueue() {
  const queryClient = useQueryClient();
  const workOrders = useQuery({ queryKey: ["work-orders"], queryFn: () => api.workOrders() });
  const requests = useQuery({
    queryKey: ["service-requests", "all"],
    queryFn: () => api.serviceRequests(),
  });
  const technicians = useQuery({ queryKey: ["technicians"], queryFn: () => api.technicians() });

  const updateStatus = useMutation({
    mutationFn: (vars: { id: number; status: WorkOrderStatus }) =>
      api.updateWorkOrderStatus(vars.id, vars.status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["work-orders"] });
      queryClient.invalidateQueries({ queryKey: ["service-requests"] });
    },
  });

  if (workOrders.isLoading) return <p className="text-slate-500">Loading work orders…</p>;
  if (workOrders.isError)
    return (
      <ErrorRetry message="Failed to load work orders." onRetry={() => workOrders.refetch()} />
    );
  if (!workOrders.data?.length)
    return <p className="text-slate-500">No work orders yet.</p>;

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left text-slate-500">
          <th className="py-2">Request</th>
          <th className="py-2">Technician</th>
          <th className="py-2">Status</th>
          <th className="py-2"></th>
        </tr>
      </thead>
      <tbody>
        {workOrders.data.map((wo) => {
          const request = requests.data?.find((r) => r.id === wo.service_request_id);
          const technician = technicians.data?.find((t) => t.id === wo.technician_id);
          const next = NEXT_STATUS[wo.status];
          const isTerminal = wo.status === "completed" || wo.status === "cancelled";
          return (
            <tr key={wo.id} className="border-b last:border-0">
              <td className="py-2 font-medium">{request?.title ?? `#${wo.service_request_id}`}</td>
              <td className="py-2">{technician?.name ?? wo.technician_id}</td>
              <td className="py-2">
                <StatusBadge status={wo.status} />
              </td>
              <td className="py-2 space-x-2">
                {next && (
                  <button
                    disabled={updateStatus.isPending}
                    onClick={() => updateStatus.mutate({ id: wo.id, status: next })}
                    className="rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white disabled:opacity-50"
                  >
                    Mark {next.replace("_", " ")}
                  </button>
                )}
                {!isTerminal && (
                  <button
                    disabled={updateStatus.isPending}
                    onClick={() => updateStatus.mutate({ id: wo.id, status: "cancelled" })}
                    className="rounded bg-slate-200 px-3 py-1 text-xs font-medium text-slate-700 disabled:opacity-50"
                  >
                    Cancel
                  </button>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function OperatorPageContent() {
  const { user } = useAuth();
  const allDepots = useQuery({ queryKey: ["depots"], queryFn: api.depots });
  const [depotId, setDepotId] = useState<number | null>(null);

  // /depots is readable by every signed-in role, but an operator is scoped to
  // one site -- offering them the other three would just hand back empty
  // tables. Narrow the filter to what the API will actually return for them.
  const depots =
    user?.role === "operator"
      ? (allDepots.data ?? []).filter((d) => d.id === user.depot_id)
      : (allDepots.data ?? []);
  const isAdmin = user?.role === "admin";

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-6xl space-y-8 p-8">
        <header className="space-y-1">
          <h1 className="text-3xl font-bold">Site Operations</h1>
          <p className="text-slate-600">
            Equipment status is kept in sync from the agent pipeline
            (fleetmind.agents.sync_equipment_status) — it is not editable here.
          </p>
        </header>

        {/* Replay rewrites predictions, decisions and equipment status across
            every site, so the API restricts it to admins. Rendering the panel
            for an operator would only produce a 403 on every poll. */}
        {isAdmin && (
          <section className="rounded-xl bg-white p-6 shadow">
            <h2 className="mb-4 text-xl font-semibold">Simulated Sensor Feed</h2>
            <ReplayControls />
          </section>
        )}

        <section className="rounded-xl bg-white p-6 shadow">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-xl font-semibold">Equipment Board</h2>
            <label className="text-sm">
              <span className="mr-2 font-medium">Site</span>
              <select
                className="rounded border bg-white px-2 py-1"
                value={depotId ?? ""}
                onChange={(e) =>
                  setDepotId(e.target.value === "" ? null : Number(e.target.value))
                }
              >
                <option value="">
                  {user?.role === "operator" ? "My site" : "All sites"}
                </option>
                {depots.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.load_hp}HP)
                  </option>
                ))}
              </select>
            </label>
          </div>
          <EquipmentBoard depotId={depotId} />
        </section>

        <section className="rounded-xl bg-white p-6 shadow">
          <div className="mb-4">
            <h2 className="text-xl font-semibold">AI-Flagged Requests</h2>
            <p className="text-sm text-slate-500">
              Auto-filed by the agent pipeline (fleetmind.agents.sync_equipment_status) when a
              HIGH or CRITICAL risk decision has no open request yet — shows the fault type, risk
              score, and decision text behind the flag, not just a status badge.
            </p>
          </div>
          <AiFlaggedPanel />
        </section>

        <section className="rounded-xl bg-white p-6 shadow">
          <h2 className="mb-4 text-xl font-semibold">Awaiting Dispatch</h2>
          <DispatchQueue />
        </section>

        <section className="rounded-xl bg-white p-6 shadow">
          <h2 className="mb-4 text-xl font-semibold">Work Order Queue</h2>
          <WorkOrderQueue />
        </section>
      </main>
    </>
  );
}

/** Signed-out visitors are bounced to /login; a role that may not open this
 *  page is sent to its own home. Usability only -- see auth-provider.tsx. */
export default function OperatorPage() {
  return (
    <RequireAuth>
      <OperatorPageContent />
    </RequireAuth>
  );
}
