"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname } from "next/navigation";
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
    return <p className="text-red-600">Failed to load depots.</p>;
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
    return <p className="text-red-600">Failed to load fault distribution.</p>;
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
  if (summary.isError) return <p className="text-red-600">Failed to load product KPIs.</p>;

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
  if (perf.isError) return <p className="text-red-600">Failed to load model performance.</p>;
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
    return <p className="text-red-600">Failed to load signal window.</p>;

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
