"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "@/lib/api";
import {
  DepotTable,
  FaultDistributionChart,
  SignalWaveform,
  StatusBanner,
} from "./components";

const FAULT_TYPES = ["Normal", "InnerRace", "OuterRace", "Ball"] as const;

function SignalViewer() {
  const [faultType, setFaultType] = useState<string>("InnerRace");
  const [experimentId, setExperimentId] = useState<number | null>(null);
  const [windowId, setWindowId] = useState<number | null>(null);

  const experiments = useQuery({
    queryKey: ["experiments", faultType],
    queryFn: () => api.experiments({ fault_type: faultType }),
  });
  const chosenExperimentId = experimentId ?? experiments.data?.[0]?.id ?? null;
  const windows = useQuery({
    queryKey: ["windows", chosenExperimentId],
    queryFn: () => api.windowsForExperiment(chosenExperimentId!, 5),
    enabled: chosenExperimentId !== null,
  });
  const chosenWindowId = windowId ?? windows.data?.[0]?.id ?? null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4">
        <label className="text-sm">
          <span className="mr-2 font-medium">Fault type</span>
          <select
            className="rounded border bg-white px-2 py-1"
            value={faultType}
            onChange={(e) => {
              setFaultType(e.target.value);
              setExperimentId(null);
              setWindowId(null);
            }}
          >
            {FAULT_TYPES.map((ft) => (
              <option key={ft}>{ft}</option>
            ))}
          </select>
        </label>
        {experiments.data && experiments.data.length > 0 && (
          <label className="text-sm">
            <span className="mr-2 font-medium">Experiment</span>
            <select
              className="rounded border bg-white px-2 py-1"
              value={chosenExperimentId ?? ""}
              onChange={(e) => {
                setExperimentId(Number(e.target.value));
                setWindowId(null);
              }}
            >
              {experiments.data.map((exp) => (
                <option key={exp.id} value={exp.id}>
                  #{exp.id} — {exp.source_file}
                </option>
              ))}
            </select>
          </label>
        )}
        {windows.data && windows.data.length > 0 && (
          <label className="text-sm">
            <span className="mr-2 font-medium">Window</span>
            <select
              className="rounded border bg-white px-2 py-1"
              value={chosenWindowId ?? ""}
              onChange={(e) => setWindowId(Number(e.target.value))}
            >
              {windows.data.map((w) => (
                <option key={w.id} value={w.id}>
                  window {w.window_index}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {experiments.data?.length === 0 && (
        <p className="text-slate-500">
          No ingested experiments with fault_type={faultType} yet.
        </p>
      )}

      {chosenWindowId !== null && <SignalWaveform windowId={chosenWindowId} />}
    </div>
  );
}

export default function DashboardPage() {
  return (
    <main className="mx-auto max-w-6xl space-y-8 p-8">
      <header className="space-y-1">
        <h1 className="text-3xl font-bold">FleetMind</h1>
        <p className="text-slate-600">
          Federated Intelligent Predictive Maintenance for Bearing Fault
          Diagnosis — CWRU dataset, 4 simulated depots (motor load levels).
        </p>
      </header>

      <StatusBanner />

      <section className="rounded-xl bg-white p-6 shadow">
        <h2 className="mb-4 text-xl font-semibold">
          Simulated Depots (real CWRU load levels)
        </h2>
        <DepotTable />
      </section>

      <section className="rounded-xl bg-white p-6 shadow">
        <h2 className="mb-4 text-xl font-semibold">Fault Class Distribution</h2>
        <FaultDistributionChart />
      </section>

      <section className="rounded-xl bg-white p-6 shadow">
        <h2 className="mb-4 text-xl font-semibold">Raw Signal Viewer</h2>
        <SignalViewer />
      </section>

      <footer className="pb-4 text-center text-xs text-slate-400">
        FleetMind · Debotri Ghosh (25MML1008) · M.Tech Project 2
      </footer>
    </main>
  );
}
