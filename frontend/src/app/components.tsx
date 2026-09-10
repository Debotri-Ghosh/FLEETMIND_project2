"use client";

import { useQuery } from "@tanstack/react-query";
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
