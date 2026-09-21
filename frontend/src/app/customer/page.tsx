"use client";

import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError, api } from "@/lib/api";
import { ErrorRetry, Nav, PriorityBadge, StatusBadge } from "../components";
import type { ServiceRequestPriority } from "@/lib/types";

function EquipmentBrowser({
  customerId,
  depotId,
  onDepotChange,
}: {
  customerId: number;
  depotId: number | null;
  onDepotChange: (depotId: number | null) => void;
}) {
  const depots = useQuery({ queryKey: ["depots"], queryFn: api.depots });
  const equipment = useQuery({
    queryKey: ["equipment", "customer", customerId, depotId],
    queryFn: () =>
      api.equipment({
        customer_id: customerId,
        depot_id: depotId ?? undefined,
      }),
  });

  return (
    <div className="space-y-4">
      <label className="text-sm">
        <span className="mr-2 font-medium">Depot</span>
        <select
          className="rounded border bg-white px-2 py-1"
          value={depotId ?? ""}
          onChange={(e) =>
            onDepotChange(e.target.value === "" ? null : Number(e.target.value))
          }
        >
          <option value="">All depots</option>
          {depots.data?.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name} ({d.load_hp}HP)
            </option>
          ))}
        </select>
      </label>

      {equipment.isLoading && <p className="text-slate-500">Loading equipment…</p>}
      {equipment.isError && (
        <ErrorRetry message="Failed to load equipment." onRetry={() => equipment.refetch()} />
      )}
      {equipment.data?.length === 0 && (
        <p className="text-slate-500">No equipment found for this customer/depot.</p>
      )}
      {equipment.data && equipment.data.length > 0 && (
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-slate-500">
              <th className="py-2">Equipment</th>
              <th className="py-2">Type</th>
              <th className="py-2">Depot</th>
              <th className="py-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {equipment.data.map((eq) => (
              <tr key={eq.id} className="border-b last:border-0">
                <td className="py-2 font-medium">{eq.name}</td>
                <td className="py-2">{eq.equipment_type}</td>
                <td className="py-2">
                  {depots.data?.find((d) => d.id === eq.depot_id)?.name ?? eq.depot_id}
                </td>
                <td className="py-2">
                  <StatusBadge status={eq.status} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function ServiceRequestForm({ customerId }: { customerId: number }) {
  const queryClient = useQueryClient();
  const equipment = useQuery({
    queryKey: ["equipment", "customer", customerId, null],
    queryFn: () => api.equipment({ customer_id: customerId }),
  });

  const [equipmentId, setEquipmentId] = useState<number | "">("");
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [priority, setPriority] = useState<ServiceRequestPriority>("medium");

  const createRequest = useMutation({
    mutationFn: api.createServiceRequest,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["service-requests", "customer", customerId] });
      setTitle("");
      setDescription("");
      setPriority("medium");
      setEquipmentId("");
    },
    onError: (error) => {
      // A 404 here almost always means the selected equipment was retired (or
      // otherwise changed) between page load and submit -- refresh the
      // equipment list so the dropdown reflects reality instead of quietly
      // failing again on retry.
      if (error instanceof ApiError && error.status === 404) {
        queryClient.invalidateQueries({ queryKey: ["equipment", "customer", customerId, null] });
      }
    },
  });

  const selectedEquipment = equipment.data?.find((eq) => eq.id === equipmentId);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedEquipment || !title.trim()) return;
    createRequest.mutate({
      customer_id: customerId,
      depot_id: selectedEquipment.depot_id,
      equipment_id: selectedEquipment.id,
      title: title.trim(),
      description: description.trim() || null,
      priority,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <label className="block text-sm">
        <span className="mb-1 block font-medium">Equipment</span>
        <select
          required
          className="w-full rounded border bg-white px-2 py-1"
          value={equipmentId}
          onChange={(e) => setEquipmentId(e.target.value === "" ? "" : Number(e.target.value))}
        >
          <option value="" disabled>
            Select equipment…
          </option>
          {equipment.data?.map((eq) => (
            <option key={eq.id} value={eq.id}>
              {eq.name} ({eq.status})
            </option>
          ))}
        </select>
      </label>

      <label className="block text-sm">
        <span className="mb-1 block font-medium">Title</span>
        <input
          required
          className="w-full rounded border bg-white px-2 py-1"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. Unusual vibration noise"
        />
      </label>

      <label className="block text-sm">
        <span className="mb-1 block font-medium">Description</span>
        <textarea
          className="w-full rounded border bg-white px-2 py-1"
          rows={3}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </label>

      <label className="block text-sm">
        <span className="mb-1 block font-medium">Priority</span>
        <select
          className="w-full rounded border bg-white px-2 py-1"
          value={priority}
          onChange={(e) => setPriority(e.target.value as ServiceRequestPriority)}
        >
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
          <option value="critical">Critical</option>
        </select>
      </label>

      <button
        type="submit"
        disabled={createRequest.isPending || !equipmentId || !title.trim()}
        className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
      >
        {createRequest.isPending ? "Submitting…" : "File service request"}
      </button>
      {createRequest.isError && (
        <p className="text-sm text-red-600">
          {createRequest.error instanceof ApiError && createRequest.error.status === 404
            ? `This equipment is no longer available (${createRequest.error.message}). The equipment list has been refreshed -- please re-select and try again.`
            : createRequest.error instanceof Error
              ? `Failed to submit request: ${createRequest.error.message}`
              : "Failed to submit request. Try again."}
        </p>
      )}
    </form>
  );
}

function ServiceRequestTracker({ customerId }: { customerId: number }) {
  const requests = useQuery({
    queryKey: ["service-requests", "customer", customerId],
    queryFn: () => api.serviceRequests({ customer_id: customerId }),
  });

  if (requests.isLoading) return <p className="text-slate-500">Loading requests…</p>;
  if (requests.isError)
    return (
      <ErrorRetry message="Failed to load requests." onRetry={() => requests.refetch()} />
    );
  if (!requests.data?.length)
    return <p className="text-slate-500">No service requests filed yet.</p>;

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left text-slate-500">
          <th className="py-2">Title</th>
          <th className="py-2">Priority</th>
          <th className="py-2">Status</th>
          <th className="py-2">Filed</th>
        </tr>
      </thead>
      <tbody>
        {requests.data.map((r) => (
          <tr key={r.id} className="border-b last:border-0">
            <td className="py-2 font-medium">{r.title}</td>
            <td className="py-2">
              <PriorityBadge priority={r.priority} />
            </td>
            <td className="py-2">
              <StatusBadge status={r.status} />
            </td>
            <td className="py-2 text-slate-500">
              {new Date(r.created_at).toLocaleString()}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function CustomerPage() {
  const customers = useQuery({ queryKey: ["customers"], queryFn: api.customers });
  const [customerId, setCustomerId] = useState<number | null>(null);
  const [depotId, setDepotId] = useState<number | null>(null);
  const activeCustomerId = customerId ?? customers.data?.[0]?.id ?? null;

  return (
    <>
      <Nav />
      <main className="mx-auto max-w-6xl space-y-8 p-8">
        <header className="space-y-1">
          <h1 className="text-3xl font-bold">Customer Portal</h1>
          <p className="text-slate-600">
            Browse your equipment and file/track service requests. No login
            yet — pick a customer to simulate their view.
          </p>
        </header>

        {customers.isLoading && <p className="text-slate-500">Loading customers…</p>}
        {customers.isError && (
          <ErrorRetry message="Failed to load customers." onRetry={() => customers.refetch()} />
        )}
        {customers.isSuccess && customers.data.length === 0 && (
          <p className="text-slate-500">No customers seeded yet.</p>
        )}

        {customers.isSuccess && customers.data.length > 0 && (
          <label className="block text-sm">
            <span className="mr-2 font-medium">Viewing as</span>
            <select
              className="rounded border bg-white px-2 py-1"
              value={activeCustomerId ?? ""}
              onChange={(e) => setCustomerId(Number(e.target.value))}
            >
              {customers.data.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} {c.company ? `— ${c.company}` : ""}
                </option>
              ))}
            </select>
          </label>
        )}

        {activeCustomerId !== null && (
          <>
            <section className="rounded-xl bg-white p-6 shadow">
              <h2 className="mb-4 text-xl font-semibold">Your Equipment</h2>
              <EquipmentBrowser
                customerId={activeCustomerId}
                depotId={depotId}
                onDepotChange={setDepotId}
              />
            </section>

            <section className="rounded-xl bg-white p-6 shadow">
              <h2 className="mb-4 text-xl font-semibold">File a Service Request</h2>
              <ServiceRequestForm customerId={activeCustomerId} />
            </section>

            <section className="rounded-xl bg-white p-6 shadow">
              <h2 className="mb-4 text-xl font-semibold">Your Service Requests</h2>
              <ServiceRequestTracker customerId={activeCustomerId} />
            </section>
          </>
        )}
      </main>
    </>
  );
}
