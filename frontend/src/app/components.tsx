"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";

const NAV_LINKS = [
  { href: "/", label: "Dashboard" },
  { href: "/customer", label: "Customer" },
  { href: "/operator", label: "Depot Operator" },
] as const;

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="border-b bg-white">
      <div className="mx-auto flex max-w-6xl items-center gap-6 px-8 py-3">
        <span className="font-bold">FleetMind</span>
        <div className="flex gap-4 text-sm">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={
                pathname === link.href
                  ? "font-semibold text-blue-600"
                  : "text-slate-600 hover:text-slate-900"
              }
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}

const STATUS_STYLES: Record<string, string> = {
  active: "bg-green-100 text-green-800",
  available: "bg-green-100 text-green-800",
  maintenance: "bg-amber-100 text-amber-800",
  busy: "bg-amber-100 text-amber-800",
  dispatched: "bg-amber-100 text-amber-800",
  traveling: "bg-amber-100 text-amber-800",
  in_progress: "bg-blue-100 text-blue-800",
  retired: "bg-slate-200 text-slate-600",
  offline: "bg-slate-200 text-slate-600",
  cancelled: "bg-slate-200 text-slate-600",
  resolved: "bg-green-100 text-green-800",
  completed: "bg-green-100 text-green-800",
  reported: "bg-red-100 text-red-800",
  diagnosed: "bg-red-100 text-red-800",
};

const PRIORITY_STYLES: Record<string, string> = {
  low: "bg-slate-200 text-slate-600",
  medium: "bg-amber-100 text-amber-800",
  high: "bg-orange-100 text-orange-800",
  critical: "bg-red-100 text-red-800",
};

export function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLES[status] ?? "bg-slate-200 text-slate-600";
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${style}`}>
      {status.replace("_", " ")}
    </span>
  );
}

export function PriorityBadge({ priority }: { priority: string }) {
  const style = PRIORITY_STYLES[priority] ?? "bg-slate-200 text-slate-600";
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${style}`}>
      {priority}
    </span>
  );
}

/** Shared error state for a page-critical query: shows the failure and lets
 * the viewer retry without a full page reload (a transient network blip on a
 * demo laptop shouldn't require one). */
export function ErrorRetry({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <p className="text-red-600">{message}</p>
      <button
        type="button"
        onClick={() => onRetry()}
        className="rounded border border-red-300 bg-white px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50"
      >
        Retry
      </button>
    </div>
  );
}

/** Replay controls: drive the simulated sensor feed.
 *
 * The underlying data is an archived CWRU dataset being replayed, NOT live
 * telemetry, and the panel says so on screen — same standard the simulated
 * GPS fields are held to elsewhere in this UI. */
export function ReplayControls() {
  const queryClient = useQueryClient();
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const status = useQuery({
    queryKey: ["replay-status"],
    queryFn: api.replayStatus,
    // Poll while the worker is ticking so progress moves on screen by itself.
    refetchInterval: (query) =>
      query.state.data?.worker.running ? 1000 : false,
  });

  // Everything the replay touches is stale the moment a tick lands.
  const refreshAll = () => {
    for (const key of [
      "equipment",
      "service-requests",
      "work-orders",
      "product-summary",
      "replay-status",
    ]) {
      queryClient.invalidateQueries({ queryKey: [key] });
    }
  };

  // While the worker runs autonomously, nothing else on the page knows a tick
  // landed — so refresh the data views each time the tick counter advances.
  const ticks = status.data?.worker.ticks ?? 0;
  useEffect(() => {
    if (ticks > 0) refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticks]);

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label);
    setNote(null);
    try {
      const result = (await fn()) as { error?: string | null } | undefined;
      if (result && typeof result === "object" && "error" in result && result.error) {
        setNote(`Tick failed: ${result.error}`);
      }
      refreshAll();
    } catch (e) {
      setNote(e instanceof Error ? e.message : "Replay command failed.");
    } finally {
      setBusy(null);
    }
  };

  if (status.isLoading)
    return <p className="text-slate-500">Loading replay status…</p>;
  if (status.isError)
    return (
      <ErrorRetry
        message="Failed to load replay status."
        onRetry={() => status.refetch()}
      />
    );

  const data = status.data;
  const running = data?.worker.running ?? false;
  const percent = data?.percent_released ?? 0;

  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-500">
        Replays the archived CWRU recordings as if sensors were streaming them in, so the
        pipeline runs incrementally on arrival. This is a replay of recorded data — it is not
        live telemetry, and no sensors are attached.
      </p>

      <div>
        <div className="mb-1 flex items-baseline justify-between text-sm">
          <span className="font-medium">
            {data?.released ?? 0} / {data?.total_windows ?? 0} windows arrived
          </span>
          <span className="text-slate-500">
            {percent}%{data?.pending ? ` · ${data.pending} awaiting processing` : ""}
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              running ? "bg-blue-600" : "bg-slate-400"
            }`}
            style={{ width: `${Math.min(percent, 100)}%` }}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={busy !== null || running || data?.complete}
          onClick={() => run("start", () => api.replayStart({ interval_seconds: 2, batch_size: 25 }))}
          className="rounded bg-blue-600 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
        >
          {busy === "start" ? "Starting…" : "Start live replay"}
        </button>
        <button
          type="button"
          disabled={busy !== null || !running}
          onClick={() => run("stop", api.replayStop)}
          className="rounded bg-slate-700 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
        >
          {busy === "stop" ? "Stopping…" : "Pause"}
        </button>
        <button
          type="button"
          disabled={busy !== null || running || data?.complete}
          onClick={() => run("step", api.replayStep)}
          className="rounded border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 disabled:opacity-50"
        >
          {busy === "step" ? "Processing…" : "Step one batch"}
        </button>
        <button
          type="button"
          disabled={busy !== null}
          onClick={() => run("reset", api.replayReset)}
          className="rounded border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 disabled:opacity-50"
        >
          {busy === "reset" ? "Resetting…" : "Reset"}
        </button>
        {running && (
          <span className="text-xs font-medium text-blue-700">
            ● live — tick {data?.worker.ticks ?? 0}
          </span>
        )}
        {data?.complete && !running && (
          <span className="text-xs font-medium text-green-700">archive fully replayed</span>
        )}
      </div>

      {data?.worker.last_tick && (
        <p className="text-xs text-slate-500">
          Last tick: {data.worker.last_tick.released} arrived ·{" "}
          {data.worker.last_tick.predictions_written} predicted ·{" "}
          {data.worker.last_tick.decisions_written} decisions ·{" "}
          {data.worker.last_tick.equipment_status_changes} equipment status changes ·{" "}
          {data.worker.last_tick.service_requests_filed} requests filed
        </p>
      )}
      {(note || data?.worker.last_error) && (
        <p className="text-xs text-red-600">{note ?? data?.worker.last_error}</p>
      )}
    </div>
  );
}

export function StatusBanner() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  if (health.isLoading) {
    return (
      <div className="rounded-lg bg-slate-200 p-4 text-slate-600">
        Connecting to FleetMind API…
      </div>
    );
  }
  if (health.isError) {
    return (
      <div className="rounded-lg bg-red-100 p-4 text-red-800">
        Cannot reach the FleetMind API. Start it with:{" "}
        <code>docker compose up api</code>
      </div>
    );
  }
  return (
    <div className="rounded-lg bg-green-100 p-4 text-green-800">
      Backend connected — database {health.data?.database} (v
      {health.data?.version})
    </div>
  );
}

export function DepotTable() {
  const depots = useQuery({ queryKey: ["depots"], queryFn: api.depots });
  if (depots.isLoading) return <p className="text-slate-500">Loading depots…</p>;
  if (depots.isError)
    return <ErrorRetry message="Failed to load depots." onRetry={() => depots.refetch()} />;
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left text-slate-500">
          <th className="py-2">Depot</th>
          <th className="py-2">Motor Load (HP)</th>
          <th className="py-2">Nominal RPM</th>
          <th className="py-2">Experiments</th>
        </tr>
      </thead>
      <tbody>
        {depots.data?.map((d) => (
          <tr key={d.id} className="border-b last:border-0">
            <td className="py-2 font-medium">{d.name}</td>
            <td className="py-2">{d.load_hp}</td>
            <td className="py-2">{d.nominal_rpm ?? "—"}</td>
            <td className="py-2">{d.n_experiments}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function FaultDistributionChart() {
  const dist = useQuery({
    queryKey: ["fault-distribution"],
    queryFn: api.faultDistribution,
  });
  if (dist.isLoading)
    return <p className="text-slate-500">Loading distribution…</p>;
  if (dist.isError)
    return (
      <ErrorRetry message="Failed to load fault distribution." onRetry={() => dist.refetch()} />
    );
  if (!dist.data?.length)
    return (
      <p className="text-slate-500">
        No signal windows ingested yet — run{" "}
        <code>python -m fleetmind.ingestion.ingest</code>.
      </p>
    );
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={dist.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="label" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="n_windows" name="Signal windows" fill="#2563eb" />
      </BarChart>
    </ResponsiveContainer>
  );
}

function StatTile({
  label,
  value,
  accent,
}: {
  label: string;
  value: number | string;
  accent?: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 p-4">
      <p className="text-sm text-slate-500">{label}</p>
      <p className={`text-2xl font-semibold ${accent ?? "text-slate-900"}`}>{value}</p>
    </div>
  );
}

export function ProductKpiRow() {
  const summary = useQuery({ queryKey: ["product-summary"], queryFn: api.productSummary });

  if (summary.isLoading) return <p className="text-slate-500">Loading product KPIs…</p>;
  if (summary.isError)
    return (
      <ErrorRetry message="Failed to load product KPIs." onRetry={() => summary.refetch()} />
    );

  const eq = summary.data?.equipment_by_status ?? {};
  const openByPriority = summary.data?.open_requests_by_priority ?? {};
  const totalOpenRequests = Object.values(openByPriority).reduce((a, b) => a + b, 0);

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
      <StatTile label="Active Equipment" value={eq.active ?? 0} />
      <StatTile
        label="Flagged for Maintenance"
        value={eq.maintenance ?? 0}
        accent={eq.maintenance ? "text-amber-700" : undefined}
      />
      <StatTile label="Open Service Requests" value={totalOpenRequests} />
      <StatTile
        label="AI-Flagged (open)"
        value={summary.data?.ai_flagged_open ?? 0}
        accent={summary.data?.ai_flagged_open ? "text-orange-700" : undefined}
      />
      <StatTile label="Active Work Orders" value={summary.data?.active_work_orders ?? 0} />
      <StatTile
        label="Critical Priority (open)"
        value={openByPriority.critical ?? 0}
        accent={openByPriority.critical ? "text-red-700" : undefined}
      />
    </div>
  );
}

export function ModelAccuracyChart() {
  const perf = useQuery({ queryKey: ["model-performance"], queryFn: api.modelPerformance });

  if (perf.isLoading) return <p className="text-slate-500">Loading model performance…</p>;
  if (perf.isError)
    return (
      <ErrorRetry message="Failed to load model performance." onRetry={() => perf.refetch()} />
    );
  if (!perf.data?.per_depot.length)
    return (
      <p className="text-slate-500">
        No training runs found in mlruns/ yet — run <code>python -m fleetmind.ml.train</code>.
      </p>
    );

  const federated = perf.data.federated;
  const chartData = perf.data.per_depot.map((d) => ({
    name: d.depot_name,
    "Local baseline": Math.round(d.best_val_acc * 1000) / 10,
    ...(federated
      ? {
          "Federated (FedProx)":
            Math.round(
              (federated.per_depot_final_accuracy[String(d.depot_id)] ?? 0) * 1000
            ) / 10,
        }
      : {}),
  }));

  return (
    <div className="space-y-3">
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e1e0d9" />
          <XAxis dataKey="name" />
          <YAxis unit="%" domain={[0, 100]} />
          <Tooltip formatter={(v: number) => `${v}%`} />
          <Legend />
          <Bar dataKey="Local baseline" fill="#2563eb" radius={[4, 4, 0, 0]} />
          {federated && (
            <Bar dataKey="Federated (FedProx)" fill="#eb6834" radius={[4, 4, 0, 0]} />
          )}
        </BarChart>
      </ResponsiveContainer>
      {federated && (
        <p className="text-sm text-slate-500">
          {federated.strategy.toUpperCase()} over {federated.rounds} rounds: mean local baseline{" "}
          {(federated.mean_baseline_accuracy * 100).toFixed(1)}% vs. mean federated{" "}
          {(federated.mean_final_accuracy * 100).toFixed(1)}% — a real, documented gap from
          federated averaging under per-depot class imbalance, not a bug.
        </p>
      )}
    </div>
  );
}

export function SignalWaveform({ windowId }: { windowId: number }) {
  const windowDetail = useQuery({
    queryKey: ["signal-window", windowId],
    queryFn: () => api.signalWindow(windowId),
  });
  if (windowDetail.isLoading)
    return <p className="text-slate-500">Loading signal…</p>;
  if (windowDetail.isError)
    return (
      <ErrorRetry
        message="Failed to load signal window."
        onRetry={() => windowDetail.refetch()}
      />
    );

  const chartData =
    windowDetail.data?.raw_values.map((v, i) => ({
      sample: i,
      amplitude: v,
    })) ?? [];

  if (chartData.length === 0) return null;
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="sample" />
        <YAxis />
        <Tooltip />
        <Line dataKey="amplitude" stroke="#2563eb" dot={false} strokeWidth={1} />
      </LineChart>
    </ResponsiveContainer>
  );
}
