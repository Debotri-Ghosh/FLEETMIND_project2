"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
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
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ApiError, api } from "@/lib/api";
import { useAuth } from "./auth-provider";

const NAV_LINKS = [
  { href: "/", label: "Dashboard", roles: ["customer", "operator", "admin"] },
  { href: "/customer", label: "Customer", roles: ["customer", "admin"] },
  { href: "/operator", label: "Site Operations", roles: ["operator", "admin"] },
] as const;

const ROLE_LABEL: Record<string, string> = {
  customer: "Customer",
  operator: "Site operator",
  admin: "Admin",
};

export function Nav() {
  const pathname = usePathname();
  const { user, signOut } = useAuth();

  // Hiding a link the caller may not use is a courtesy, not a control: the
  // API refuses the underlying requests regardless of what this renders.
  const links = NAV_LINKS.filter(
    (link) => user && (link.roles as readonly string[]).includes(user.role)
  );

  return (
    <nav className="border-b bg-white">
      <div className="mx-auto flex max-w-6xl items-center gap-6 px-8 py-3">
        <span className="font-bold">FleetMind</span>
        <div className="flex gap-4 text-sm">
          {links.map((link) => (
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

        {user && (
          <div className="ml-auto flex items-center gap-3 text-sm">
            <span className="text-slate-600">
              <span className="font-medium">{ROLE_LABEL[user.role] ?? user.role}</span>
              {user.site_name && (
                <span className="text-slate-500"> · {user.site_name}</span>
              )}
              {user.customer_name && (
                <span className="text-slate-500"> · {user.customer_name}</span>
              )}
            </span>
            <button
              type="button"
              onClick={signOut}
              className="rounded border border-slate-300 px-2 py-1 text-xs font-medium hover:bg-slate-100"
            >
              Sign out
            </button>
          </div>
        )}
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

/** The evidence behind one AI-filed request: the signal it was decided from,
 * its spectrum with the corroborating bearing defect frequency marked, the
 * model's confidence, and the diagnostic agent's own statistics.
 *
 * The point is that a reviewer can *check* the finding instead of trusting
 * it -- so the copy is careful about what the band does and doesn't mean. */
export function EvidencePanel({ requestId }: { requestId: number }) {
  const evidence = useQuery({
    queryKey: ["evidence", requestId],
    queryFn: () => api.serviceRequestEvidence(requestId),
  });

  if (evidence.isLoading) return <p className="text-slate-500">Loading evidence…</p>;
  if (evidence.isError)
    return (
      <ErrorRetry
        message={
          evidence.error instanceof ApiError
            ? evidence.error.message
            : "Failed to load evidence."
        }
        onRetry={() => evidence.refetch()}
      />
    );

  const e = evidence.data;
  if (!e) return null;

  const waveform = e.raw_values.map((v, i) => ({
    ms: Number(((i / e.sampling_rate_hz) * 1000).toFixed(3)),
    amplitude: v,
  }));
  const spectrumData = e.spectrum_freqs_hz.map((hz, i) => ({
    hz,
    magnitude: e.spectrum_magnitudes[i],
  }));
  const band = e.matched_frequency;

  return (
    <div className="space-y-5 rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
        <span>
          <span className="text-slate-500">Signal window</span>{" "}
          <span className="font-mono font-medium">#{e.window_id}</span>
        </span>
        <span>
          <span className="text-slate-500">Site</span>{" "}
          <span className="font-medium">{e.site_name ?? "—"}</span>
        </span>
        <span>
          <span className="text-slate-500">Model</span>{" "}
          <span className="font-medium">{e.predicted_label}</span>
          {e.model_confidence !== null && (
            <span className="text-slate-500">
              {" "}
              at {(e.model_confidence * 100).toFixed(1)}% confidence
            </span>
          )}
        </span>
        {e.model_version && (
          <span>
            <span className="text-slate-500">Checkpoint</span>{" "}
            <span className="font-mono text-xs">{e.model_version}</span>
          </span>
        )}
      </div>

      <div>
        <h4 className="mb-1 text-sm font-semibold">Raw vibration signal</h4>
        <p className="mb-2 text-xs text-slate-500">
          The exact window this decision was made from — {e.raw_values.length} samples at{" "}
          {e.sampling_rate_hz.toLocaleString()} Hz.
        </p>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={waveform}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e1e0d9" />
            <XAxis
              dataKey="ms"
              tick={{ fontSize: 11 }}
              label={{ value: "ms", position: "insideBottomRight", offset: -4, fontSize: 11 }}
            />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v: number) => v.toFixed(4)} />
            <Line dataKey="amplitude" stroke="#2563eb" dot={false} strokeWidth={1} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h4 className="mb-1 text-sm font-semibold">Frequency spectrum</h4>
        <p className="mb-2 text-xs text-slate-500">
          {band ? (
            <>
              The shaded band is <span className="font-medium">{band.name}</span> ={" "}
              {band.hz} Hz ({band.band_low_hz}–{band.band_high_hz} Hz), the frequency a{" "}
              {e.predicted_label} defect produces at this site&apos;s shaft speed. Energy sitting
              on that band is independent physical corroboration of the call — it is{" "}
              <span className="font-medium">not</span> what the model looked at; the CNN reads a
              log-scaled, normalised spectrum and was never given these frequencies.
            </>
          ) : (
            <>
              No defect band is drawn: {e.predicted_label} has no characteristic fault frequency
              to corroborate against.
            </>
          )}
        </p>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={spectrumData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e1e0d9" />
            <XAxis
              dataKey="hz"
              type="number"
              domain={[0, "dataMax"]}
              tick={{ fontSize: 11 }}
              label={{ value: "Hz", position: "insideBottomRight", offset: -4, fontSize: 11 }}
            />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip
              formatter={(v: number) => v.toFixed(3)}
              labelFormatter={(hz) => `${hz} Hz`}
            />
            {band && (
              <ReferenceArea
                x1={band.band_low_hz}
                x2={band.band_high_hz}
                fill="#eb6834"
                fillOpacity={0.18}
              />
            )}
            {band && (
              <ReferenceLine
                x={band.hz}
                stroke="#c85a28"
                strokeDasharray="4 3"
                label={{ value: band.name, position: "top", fontSize: 11, fill: "#c85a28" }}
              />
            )}
            <Line dataKey="magnitude" stroke="#2563eb" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
        <p className="mt-1 text-xs text-slate-500">
          All defect frequencies at this shaft speed:{" "}
          {Object.entries(e.all_characteristic_frequencies)
            .map(([k, v]) => `${k} ${v} Hz`)
            .join(" · ")}
        </p>
      </div>

      <div>
        <h4 className="mb-1 text-sm font-semibold">Diagnostic agent statistics</h4>
        <p className="mb-2 text-xs text-slate-500">
          Recomputed from the window above with the same function that produced the decision
          (only the decision text and risk score are stored).
        </p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile label="Signal RMS" value={e.signal_rms.toFixed(4)} />
          <StatTile label="Peak-to-peak" value={e.signal_peak.toFixed(4)} />
          <StatTile
            label="Diagnosis confidence"
            value={`${(e.diagnosis_confidence * 100).toFixed(0)}%`}
          />
          <StatTile
            label="Risk score"
            value={e.risk_score !== null ? `${(e.risk_score * 100).toFixed(0)}%` : "—"}
            accent={e.risk_score !== null && e.risk_score >= 0.7 ? "text-red-700" : undefined}
          />
        </div>
        <p className="mt-2 text-xs text-slate-600">
          <span className="text-slate-500">Diagnosis:</span> {e.diagnosis}
        </p>
      </div>

      <NarrativePanel requestId={requestId} />
    </div>
  );
}

const CORROBORATION_STYLE: Record<string, string> = {
  strong: "bg-green-100 text-green-800",
  partial: "bg-amber-100 text-amber-800",
  weak: "bg-orange-100 text-orange-800",
  none: "bg-slate-200 text-slate-700",
};

/** Claude's write-up of one AI-filed request.
 *
 * Generation is an explicit button press, not an automatic fetch on expand:
 * it costs a real API call. Once generated the result is cached server-side,
 * so reopening the panel is free. */
export function NarrativePanel({ requestId }: { requestId: number }) {
  const queryClient = useQueryClient();
  const cached = useQuery({
    queryKey: ["narrative", requestId],
    queryFn: () => api.serviceRequestNarrative(requestId),
    // A 404 here means "not generated yet", which is a normal state, not a
    // failure worth retrying.
    retry: false,
  });

  const generate = useMutation({
    mutationFn: (force: boolean) => api.generateServiceRequestNarrative(requestId, force),
    onSuccess: (data) => queryClient.setQueryData(["narrative", requestId], data),
  });

  const narrative = generate.data ?? cached.data;
  const notYetGenerated =
    cached.isError && cached.error instanceof ApiError && cached.error.status === 404;

  if (cached.isLoading) return <p className="text-xs text-slate-500">Checking for a write-up…</p>;

  if (cached.isError && !notYetGenerated && !narrative) {
    return (
      <ErrorRetry
        message={
          cached.error instanceof ApiError ? cached.error.message : "Failed to load write-up."
        }
        onRetry={() => cached.refetch()}
      />
    );
  }

  return (
    <div>
      <div className="mb-1 flex flex-wrap items-center gap-3">
        <h4 className="text-sm font-semibold">Engineer&apos;s write-up (Claude)</h4>
        {narrative && (
          <span
            className={`rounded px-2 py-0.5 text-xs font-medium ${
              CORROBORATION_STYLE[narrative.corroboration] ?? CORROBORATION_STYLE.none
            }`}
          >
            {narrative.corroboration} corroboration
          </span>
        )}
        <button
          type="button"
          onClick={() => generate.mutate(Boolean(narrative))}
          disabled={generate.isPending}
          className="rounded border border-slate-300 bg-white px-2 py-1 text-xs font-medium hover:bg-slate-100 disabled:opacity-50"
        >
          {generate.isPending
            ? "Generating…"
            : narrative
              ? "Regenerate"
              : "Generate write-up"}
        </button>
      </div>

      <p className="mb-2 text-xs text-slate-500">
        A language model wrote this prose by reasoning over the same deterministic functions
        shown above, which it called as tools. It supplies{" "}
        <span className="font-medium">no numbers of its own</span> — every figure came back from
        a tool, and anything untraceable is listed below. The risk score and urgency are
        unchanged by it, and a human still decides.
      </p>

      {generate.isError && (
        <p className="mb-2 rounded bg-red-50 p-2 text-xs text-red-700">
          {generate.error instanceof ApiError
            ? generate.error.message
            : "Write-up generation failed."}
        </p>
      )}

      {!narrative && !generate.isPending && (
        <p className="text-xs text-slate-500">
          No write-up generated yet for this request.
        </p>
      )}

      {narrative && (
        <div className="space-y-3 rounded border border-slate-200 bg-white p-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Diagnosis
            </p>
            <p className="text-sm text-slate-800">{narrative.diagnosis_narrative}</p>
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Recommendation
            </p>
            <p className="text-sm text-slate-800">{narrative.maintenance_recommendation}</p>
          </div>
          {narrative.evidence_cited.length > 0 && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Figures relied on
              </p>
              <ul className="list-inside list-disc text-xs text-slate-600">
                {narrative.evidence_cited.map((item) => (
                  <li key={item} className="font-mono">
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {narrative.ungrounded_numbers.length > 0 && (
            <p className="rounded bg-amber-50 p-2 text-xs text-amber-800">
              Grounding check: {narrative.ungrounded_numbers.join(", ")} could not be traced to
              any tool result. Treat those figures as unverified.
            </p>
          )}
          <details className="text-xs text-slate-600">
            <summary className="cursor-pointer text-slate-500">
              Tool calls behind this write-up ({narrative.tool_calls.length})
            </summary>
            <ul className="mt-2 space-y-2">
              {narrative.tool_calls.map((call, i) => (
                <li key={`${call.tool}-${i}`}>
                  <span className="font-mono font-medium">{call.tool}</span>
                  {Object.keys(call.input).length > 0 && (
                    <span className="font-mono"> {JSON.stringify(call.input)}</span>
                  )}
                  <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-all rounded bg-slate-50 p-2 font-mono text-[11px]">
                    {call.result}
                  </pre>
                </li>
              ))}
            </ul>
          </details>
          <p className="text-xs text-slate-400">
            {narrative.model} · {narrative.input_tokens.toLocaleString()} in /{" "}
            {narrative.output_tokens.toLocaleString()} out tokens ·{" "}
            {narrative.cached ? "cached" : "generated just now"}
          </p>
        </div>
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
  if (depots.isLoading) return <p className="text-slate-500">Loading sites…</p>;
  if (depots.isError)
    return <ErrorRetry message="Failed to load sites." onRetry={() => depots.refetch()} />;
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left text-slate-500">
          <th className="py-2">Site</th>
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
          federated averaging under per-site class imbalance, not a bug.
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
