"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "@/lib/api";
import { RequireAuth } from "./auth-provider";
import {
  DepotTable,
  FaultDistributionChart,
  ModelAccuracyChart,
  Nav,
  ProductKpiRow,
  StatusBanner,
} from "./components";
import { FrequencyLab } from "./frequency-lab";
import { LiveAutomationBoard } from "./live-board";
import { Panel, PanelHeader, SectionLabel } from "./ui";

const FAULT_TYPES = ["Normal", "Ball", "InnerRace", "OuterRace"] as const;

function Select({
  label,
  value,
  onChange,
  children,
}: {
  label: string;
  value: string | number;
  onChange: (v: string) => void;
  children: React.ReactNode;
}) {
  return (
    <label className="flex items-center gap-2 text-sm">
      <SectionLabel>{label}</SectionLabel>
      <select
        className="rounded border border-hair-strong bg-panel-raised px-2 py-1 text-sm text-ink"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {children}
      </select>
    </label>
  );
}

/** Picks a window, then hands it to the Frequency Lab. */
function SignalExplorer() {
  const [faultType, setFaultType] = useState<string>("Ball");
  const [experimentId, setExperimentId] = useState<number | null>(null);
  const [windowId, setWindowId] = useState<number | null>(null);

  const experiments = useQuery({
    queryKey: ["experiments", faultType],
    queryFn: () => api.experiments({ fault_type: faultType }),
  });
  const chosenExperimentId = experimentId ?? experiments.data?.[0]?.id ?? null;

  const windows = useQuery({
    queryKey: ["windows", chosenExperimentId],
    queryFn: () => api.windowsForExperiment(chosenExperimentId!, 8),
    enabled: chosenExperimentId !== null,
  });
  const chosenWindowId = windowId ?? windows.data?.[0]?.id ?? null;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <Select
          label="Fault class"
          value={faultType}
          onChange={(v) => {
            setFaultType(v);
            setExperimentId(null);
            setWindowId(null);
          }}
        >
          {FAULT_TYPES.map((ft) => (
            <option key={ft}>{ft}</option>
          ))}
        </Select>

        {experiments.data && experiments.data.length > 0 && (
          <Select
            label="Recording"
            value={chosenExperimentId ?? ""}
            onChange={(v) => {
              setExperimentId(Number(v));
              setWindowId(null);
            }}
          >
            {experiments.data.map((exp) => (
              <option key={exp.id} value={exp.id}>
                #{exp.id} — {exp.source_file}
              </option>
            ))}
          </Select>
        )}

        {windows.data && windows.data.length > 0 && (
          <Select
            label="Window"
            value={chosenWindowId ?? ""}
            onChange={(v) => setWindowId(Number(v))}
          >
            {windows.data.map((w) => (
              <option key={w.id} value={w.id}>
                window {w.window_index}
              </option>
            ))}
          </Select>
        )}
      </div>

      {experiments.data?.length === 0 && (
        <p className="text-sm text-ink-3">
          No ingested recordings with fault class {faultType} yet.
        </p>
      )}

      {chosenWindowId !== null && <FrequencyLab windowId={chosenWindowId} />}
    </div>
  );
}

function DashboardPageContent() {
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-6xl space-y-6 px-6 py-8">
        <header className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight text-ink">FleetMind</h1>
          <p className="max-w-3xl text-sm text-ink-2">
            Federated predictive maintenance for rotating industrial machinery — CWRU
            bearing dataset, four real motor load levels standing in for four
            manufacturing sites.
          </p>
        </header>

        {/* The centrepiece: automation you can watch happen. */}
        <LiveAutomationBoard />

        <StatusBanner />

        <Panel>
          <PanelHeader
            title="Frequency Lab"
            hint="Pick any recorded window, then transform it yourself. The frequency view is the same samples re-expressed — not a second measurement."
          />
          <div className="px-5 py-5">
            <SignalExplorer />
          </div>
        </Panel>

        <Panel>
          <PanelHeader title="System Overview" />
          <div className="px-5 py-5">
            <ProductKpiRow />
          </div>
        </Panel>

        <Panel>
          <PanelHeader
            title="Model Performance: Local vs. Federated"
            hint="Honest, experiment-level-split results — each site's own classifier against the same architecture trained with FedProx across all four sites."
          />
          <div className="px-5 py-5">
            <ModelAccuracyChart />
          </div>
        </Panel>

        <div className="grid gap-6 lg:grid-cols-2">
          <Panel>
            <PanelHeader title="Simulated Sites" hint="Real CWRU motor load levels." />
            <div className="px-5 py-5">
              <DepotTable />
            </div>
          </Panel>

          <Panel>
            <PanelHeader title="Fault Class Distribution" />
            <div className="px-5 py-5">
              <FaultDistributionChart />
            </div>
          </Panel>
        </div>

        <footer className="pb-2 text-center text-xs text-ink-3">
          FleetMind · Debotri Ghosh (25MML1008) · M.Tech Project 2 · CWRU Bearing Data
          Center
        </footer>
      </main>
    </>
  );
}

/** Signed-out visitors are bounced to /login; a role that may not open this
 *  page is sent to its own home. Usability only -- see auth-provider.tsx. */
export default function DashboardPage() {
  return (
    <RequireAuth>
      <DashboardPageContent />
    </RequireAuth>
  );
}
