"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  LabelList,
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
import {
  ACCENT,
  AXIS_TICK,
  COMPARE,
  INK,
  TOOLTIP_STYLE,
  faultColor,
} from "@/lib/theme";
import { useAuth } from "./auth-provider";
import { GroundTruthVsPrediction } from "./ground-truth";
import { SectionLabel } from "./ui";

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
    <nav className="border-b border-hair bg-panel">
      <div className="mx-auto flex max-w-6xl items-center gap-6 px-8 py-3">
        <span className="font-bold">FleetMind</span>
        <div className="flex gap-4 text-sm">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={
                pathname === link.href
                  ? "font-semibold text-accent"
                  : "text-ink-2 hover:text-ink"
              }
            >
              {link.label}
            </Link>
          ))}
        </div>

        {user && (
          <div className="ml-auto flex items-center gap-3 text-sm">
            <span className="text-ink-2">
              <span className="font-medium">{ROLE_LABEL[user.role] ?? user.role}</span>
              {user.site_name && (
                <span className="text-ink-3"> · {user.site_name}</span>
              )}
              {user.customer_name && (
                <span className="text-ink-3"> · {user.customer_name}</span>
              )}
            </span>
            <button
              type="button"
              onClick={signOut}
              className="rounded border border-hair-strong px-2 py-1 text-xs font-medium hover:bg-panel-raised"
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
  active: "bg-ok/15 text-ok",
  available: "bg-ok/15 text-ok",
  maintenance: "bg-warn/15 text-warn",
  busy: "bg-warn/15 text-warn",
  dispatched: "bg-warn/15 text-warn",
  traveling: "bg-warn/15 text-warn",
  in_progress: "bg-s-2/15 text-s-2",
  retired: "bg-hair text-ink-2",
  offline: "bg-hair text-ink-2",
  cancelled: "bg-hair text-ink-2",
  resolved: "bg-ok/15 text-ok",
  completed: "bg-ok/15 text-ok",
  reported: "bg-crit/15 text-crit",
  diagnosed: "bg-crit/15 text-crit",
};

const PRIORITY_STYLES: Record<string, string> = {
  low: "bg-hair text-ink-2",
  medium: "bg-warn/15 text-warn",
  high: "bg-orange-100 text-orange-800",
  critical: "bg-crit/15 text-crit",
};

export function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLES[status] ?? "bg-hair text-ink-2";
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${style}`}>
      {status.replace("_", " ")}
    </span>
  );
}

export function PriorityBadge({ priority }: { priority: string }) {
  const style = PRIORITY_STYLES[priority] ?? "bg-hair text-ink-2";
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
      <p className="text-crit">{message}</p>
      <button
        type="button"
        onClick={() => onRetry()}
        className="rounded border border-red-300 bg-panel px-2 py-1 text-xs font-medium text-crit hover:bg-red-50"
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
    return <p className="text-ink-3">Loading replay status…</p>;
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
      <p className="text-sm text-ink-3">
        Replays the archived CWRU recordings as if sensors were streaming them in, so the
        pipeline runs incrementally on arrival. This is a replay of recorded data — it is not
        live telemetry, and no sensors are attached.
      </p>

      <div>
        <div className="mb-1 flex items-baseline justify-between text-sm">
          <span className="font-medium">
            {data?.released ?? 0} / {data?.total_windows ?? 0} windows arrived
          </span>
          <span className="text-ink-3">
            {percent}%{data?.pending ? ` · ${data.pending} awaiting processing` : ""}
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-hair">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              running ? "bg-blue-600" : "bg-hair-strong"
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
          className="rounded bg-accent px-3 py-1.5 text-xs font-medium text-canvas disabled:opacity-50"
        >
          {busy === "start" ? "Starting…" : "Start live replay"}
        </button>
        <button
          type="button"
          disabled={busy !== null || !running}
          onClick={() => run("stop", api.replayStop)}
          className="rounded bg-panel-raised px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
        >
          {busy === "stop" ? "Stopping…" : "Pause"}
        </button>
        <button
          type="button"
          disabled={busy !== null || running || data?.complete}
          onClick={() => run("step", api.replayStep)}
          className="rounded border border-hair-strong bg-panel-raised px-3 py-1.5 text-xs font-medium text-ink-2 disabled:opacity-50"
        >
          {busy === "step" ? "Processing…" : "Step one batch"}
        </button>
        <button
          type="button"
          disabled={busy !== null}
          onClick={() => run("reset", api.replayReset)}
          className="rounded border border-hair-strong bg-panel-raised px-3 py-1.5 text-xs font-medium text-ink-2 disabled:opacity-50"
        >
          {busy === "reset" ? "Resetting…" : "Reset"}
        </button>
        {running && (
          <span className="text-xs font-medium text-accent">
            ● live — tick {data?.worker.ticks ?? 0}
          </span>
        )}
        {data?.complete && !running && (
          <span className="text-xs font-medium text-ok">archive fully replayed</span>
        )}
      </div>

      {data?.worker.last_tick && (
        <p className="text-xs text-ink-3">
          Last tick: {data.worker.last_tick.released} arrived ·{" "}
          {data.worker.last_tick.predictions_written} predicted ·{" "}
          {data.worker.last_tick.decisions_written} decisions ·{" "}
          {data.worker.last_tick.equipment_status_changes} equipment status changes ·{" "}
          {data.worker.last_tick.service_requests_filed} requests filed
        </p>
      )}
      {(note || data?.worker.last_error) && (
        <p className="text-xs text-crit">{note ?? data?.worker.last_error}</p>
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

  if (evidence.isLoading) return <p className="text-ink-3">Loading evidence…</p>;
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
    <div className="space-y-5 rounded-lg border border-hair bg-panel-sunken p-4">
      {/* The first question a reviewer asks is whether the model simply read
          the label, so this answers it before anything else on the panel. */}
      <div>
        <SectionLabel>Ground truth vs. independent prediction</SectionLabel>
        <div className="mt-2">
          <GroundTruthVsPrediction evidence={e} />
        </div>
      </div>

      <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
        <span>
          <span className="text-ink-3">Signal window</span>{" "}
          <span className="font-mono font-medium">#{e.window_id}</span>
        </span>
        <span>
          <span className="text-ink-3">Site</span>{" "}
          <span className="font-medium">{e.site_name ?? "—"}</span>
        </span>
        <span>
          <span className="text-ink-3">Model</span>{" "}
          <span className="font-medium">{e.predicted_label}</span>
          {e.model_confidence !== null && (
            <span className="text-ink-3">
              {" "}
              at {(e.model_confidence * 100).toFixed(1)}% confidence
            </span>
          )}
        </span>
        {e.model_version && (
          <span>
            <span className="text-ink-3">Checkpoint</span>{" "}
            <span className="font-mono text-xs">{e.model_version}</span>
          </span>
        )}
      </div>

      <div>
        <h4 className="mb-1 text-sm font-semibold">Raw vibration signal</h4>
        <p className="mb-2 text-xs text-ink-3">
          The exact window this decision was made from — {e.raw_values.length} samples at{" "}
          {e.sampling_rate_hz.toLocaleString()} Hz.
        </p>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={waveform}>
            <CartesianGrid strokeDasharray="2 4" stroke="var(--grid)" vertical={false} />
            <XAxis
              dataKey="ms"
              tick={AXIS_TICK}
              label={{ value: "ms", position: "insideBottomRight", offset: -4, fontSize: 11, fill: INK.muted }}
            />
            <YAxis tick={AXIS_TICK} />
            <Tooltip formatter={(v: number) => v.toFixed(4)} />
            <Line dataKey="amplitude" stroke={COMPARE.a} dot={false} strokeWidth={1} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h4 className="mb-1 text-sm font-semibold">Frequency spectrum</h4>
        <p className="mb-2 text-xs text-ink-3">
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
            <CartesianGrid strokeDasharray="2 4" stroke="var(--grid)" vertical={false} />
            <XAxis
              dataKey="hz"
              type="number"
              domain={[0, "dataMax"]}
              tick={AXIS_TICK}
              label={{ value: "Hz", position: "insideBottomRight", offset: -4, fontSize: 11, fill: INK.muted }}
            />
            <YAxis tick={AXIS_TICK} />
            <Tooltip
              formatter={(v: number) => v.toFixed(3)}
              labelFormatter={(hz) => `${hz} Hz`}
            />
            {band && (
              <ReferenceArea
                x1={band.band_low_hz}
                x2={band.band_high_hz}
                fill={ACCENT}
                fillOpacity={0.18}
              />
            )}
            {band && (
              <ReferenceLine
                x={band.hz}
                stroke={ACCENT}
                strokeDasharray="4 3"
                label={{ value: band.name, position: "top", fontSize: 11, fill: ACCENT }}
              />
            )}
            <Line dataKey="magnitude" stroke={COMPARE.a} dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
        <p className="mt-1 text-xs text-ink-3">
          All defect frequencies at this shaft speed:{" "}
          {Object.entries(e.all_characteristic_frequencies)
            .map(([k, v]) => `${k} ${v} Hz`)
            .join(" · ")}
        </p>
      </div>

      <div>
        <h4 className="mb-1 text-sm font-semibold">Diagnostic agent statistics</h4>
        <p className="mb-2 text-xs text-ink-3">
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
            accent={e.risk_score !== null && e.risk_score >= 0.7 ? "text-crit" : undefined}
          />
        </div>
        <p className="mt-2 text-xs text-ink-2">
          <span className="text-ink-3">Diagnosis:</span> {e.diagnosis}
        </p>
      </div>

      <NarrativePanel requestId={requestId} />
    </div>
  );
}

const CORROBORATION_STYLE: Record<string, string> = {
  strong: "bg-ok/15 text-ok",
  partial: "bg-warn/15 text-warn",
  weak: "bg-orange-100 text-orange-800",
  none: "bg-hair text-ink-2",
};

/** The LLM's write-up of one AI-filed request.
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

  if (cached.isLoading) return <p className="text-xs text-ink-3">Checking for a write-up…</p>;

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
        <h4 className="text-sm font-semibold">Engineer&apos;s write-up (LLM)</h4>
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
          className="rounded border border-hair-strong bg-panel-raised px-2 py-1 text-xs font-medium hover:bg-panel-raised disabled:opacity-50"
        >
          {generate.isPending
            ? "Generating…"
            : narrative
              ? "Regenerate"
              : "Generate write-up"}
        </button>
      </div>

      <p className="mb-2 text-xs text-ink-3">
        A language model wrote this prose by reasoning over the same deterministic functions
        shown above, which it called as tools. It supplies{" "}
        <span className="font-medium">no numbers of its own</span> — every figure came back from
        a tool, and anything untraceable is listed below. The risk score and urgency are
        unchanged by it, and a human still decides.
      </p>

      {generate.isError && (
        <p className="mb-2 rounded bg-red-50 p-2 text-xs text-crit">
          {generate.error instanceof ApiError
            ? generate.error.message
            : "Write-up generation failed."}
        </p>
      )}

      {!narrative && !generate.isPending && (
        <p className="text-xs text-ink-3">
          No write-up generated yet for this request.
        </p>
      )}

      {narrative && (
        <div className="space-y-3 rounded border border-hair bg-panel p-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-ink-3">
              Diagnosis
            </p>
            <p className="text-sm text-ink">{narrative.diagnosis_narrative}</p>
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-ink-3">
              Recommendation
            </p>
            <p className="text-sm text-ink">{narrative.maintenance_recommendation}</p>
          </div>
          {narrative.evidence_cited.length > 0 && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-ink-3">
                Figures relied on
              </p>
              <ul className="list-inside list-disc text-xs text-ink-2">
                {narrative.evidence_cited.map((item) => (
                  <li key={item} className="font-mono">
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {narrative.ungrounded_numbers.length > 0 && (
            <p className="rounded border border-warn/40 bg-warn/10 p-2 text-xs text-warn">
              Grounding check: {narrative.ungrounded_numbers.join(", ")} could not be traced to
              any tool result. Treat those figures as unverified.
            </p>
          )}
          <details className="text-xs text-ink-2">
            <summary className="cursor-pointer text-ink-3">
              Tool calls behind this write-up ({narrative.tool_calls.length})
            </summary>
            <ul className="mt-2 space-y-2">
              {narrative.tool_calls.map((call, i) => (
                <li key={`${call.tool}-${i}`}>
                  <span className="font-mono font-medium">{call.tool}</span>
                  {Object.keys(call.input).length > 0 && (
                    <span className="font-mono"> {JSON.stringify(call.input)}</span>
                  )}
                  <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-all rounded bg-panel-sunken p-2 font-mono text-[11px]">
                    {call.result}
                  </pre>
                </li>
              ))}
            </ul>
          </details>
          <p className="text-xs text-ink-3">
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
      <div className="rounded-lg bg-hair p-4 text-ink-2">
        Connecting to FleetMind API…
      </div>
    );
  }
  if (health.isError) {
    return (
      <div className="rounded-lg border border-crit/40 bg-crit/10 p-4 text-crit">
        Cannot reach the FleetMind API. Start it with:{" "}
        <code>docker compose up api</code>
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-ok/40 bg-ok/10 p-4 text-ok">
      Backend connected — database {health.data?.database} (v
      {health.data?.version})
    </div>
  );
}

export function DepotTable() {
  const depots = useQuery({ queryKey: ["depots"], queryFn: api.depots });
  if (depots.isLoading) return <p className="text-ink-3">Loading sites…</p>;
  if (depots.isError)
    return <ErrorRetry message="Failed to load sites." onRetry={() => depots.refetch()} />;
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left text-ink-3">
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
    return <p className="text-ink-3">Loading distribution…</p>;
  if (dist.isError)
    return (
      <ErrorRetry message="Failed to load fault distribution." onRetry={() => dist.refetch()} />
    );
  if (!dist.data?.length)
    return (
      <p className="text-ink-3">
        No signal windows ingested yet — run{" "}
        <code>python -m fleetmind.ingestion.ingest</code>.
      </p>
    );
  // One hue per fault class, in the fixed validated order -- the same colour
  // this class carries everywhere else in the app, so identity is consistent
  // across the dashboard, the Frequency Lab and the evidence panel. A single
  // series is direct-labelled rather than given a legend box.
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={dist.data} margin={{ top: 16, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="label" tick={AXIS_TICK} tickLine={false} />
        <YAxis tick={AXIS_TICK} tickLine={false} width={48} />
        <Tooltip
          {...TOOLTIP_STYLE}
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
          formatter={(v: number) => [v.toLocaleString(), "signal windows"]}
        />
        <Bar dataKey="n_windows" name="Signal windows" radius={[4, 4, 0, 0]}>
          <LabelList
            dataKey="n_windows"
            position="top"
            fontSize={11}
            fill={INK.secondary}
            formatter={(v: number) => v.toLocaleString()}
          />
          {dist.data.map((entry) => (
            <Cell key={entry.label} fill={faultColor(entry.label)} />
          ))}
        </Bar>
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
    <div className="rounded-lg border border-hair p-4">
      <p className="text-sm text-ink-3">{label}</p>
      <p className={`text-2xl font-semibold ${accent ?? "text-ink"}`}>{value}</p>
    </div>
  );
}

export function ProductKpiRow() {
  const summary = useQuery({ queryKey: ["product-summary"], queryFn: api.productSummary });

  if (summary.isLoading) return <p className="text-ink-3">Loading product KPIs…</p>;
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
        accent={openByPriority.critical ? "text-crit" : undefined}
      />
    </div>
  );
}

export function ModelAccuracyChart() {
  const perf = useQuery({ queryKey: ["model-performance"], queryFn: api.modelPerformance });

  if (perf.isLoading) return <p className="text-ink-3">Loading model performance…</p>;
  if (perf.isError)
    return (
      <ErrorRetry message="Failed to load model performance." onRetry={() => perf.refetch()} />
    );
  if (!perf.data?.per_depot.length)
    return (
      <p className="text-ink-3">
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
          <CartesianGrid strokeDasharray="2 4" stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="name" />
          <YAxis unit="%" domain={[0, 100]} />
          <Tooltip formatter={(v: number) => `${v}%`} />
          <Legend />
          <Bar dataKey="Local baseline" fill={COMPARE.a} radius={[4, 4, 0, 0]} />
          {federated && (
            <Bar dataKey="Federated (FedProx)" fill={COMPARE.b} radius={[4, 4, 0, 0]} />
          )}
        </BarChart>
      </ResponsiveContainer>
      {federated && (
        <p className="text-sm text-ink-3">
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
    return <p className="text-ink-3">Loading signal…</p>;
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
        <Line dataKey="amplitude" stroke={COMPARE.a} dot={false} strokeWidth={1} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
