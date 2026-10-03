"use client";

/**
 * The live automation board -- the centrepiece of the dashboard.
 *
 * The brief this answers: someone glancing at the screen should see automation
 * happening without being told where to look. So three things move on their
 * own while the worker runs:
 *
 *   1. the arrival counter and progress bar advance;
 *   2. an equipment card flashes amber at the moment its status changes;
 *   3. a newly filed AI request animates into the list.
 *
 * (2) and (3) are the ones that matter, and they are the reason this component
 * keeps its own `useRef` snapshots of the previous equipment statuses and
 * request ids: the API returns current state, not a diff, so "what just
 * changed" has to be derived here by comparing successive responses. Without
 * that the board would still be correct and would look completely static.
 *
 * Replay *control* is admin-only at the API, so operators and customers get
 * the same live board without the buttons rather than a panel that 403s on
 * every poll.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError, api } from "@/lib/api";
import { STATUS } from "@/lib/theme";
import type { Equipment, ServiceRequest } from "@/lib/types";
import { useAuth } from "./auth-provider";
import {
  Button,
  Empty,
  LiveDot,
  Metric,
  Panel,
  ProgressBar,
  SectionLabel,
  StatusPill,
} from "./ui";

/** Keys invalidated whenever a tick lands -- everything a tick can change. */
const LIVE_KEYS = [
  "equipment",
  "service-requests",
  "work-orders",
  "product-summary",
  "replay-status",
];

export function LiveAutomationBoard() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const isAdmin = user?.role === "admin";

  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const status = useQuery({
    queryKey: ["replay-status"],
    queryFn: api.replayStatus,
    enabled: isAdmin, // /replay/* is admin-only; do not 403 everyone else
    refetchInterval: (query) => (query.state.data?.worker.running ? 1000 : false),
  });

  const running = status.data?.worker.running ?? false;

  const equipment = useQuery({
    queryKey: ["equipment", "live-board"],
    queryFn: () => api.equipment(),
    // Non-admins cannot see the worker state, so they poll gently instead of
    // waiting for a tick signal they will never receive.
    refetchInterval: running ? 1200 : isAdmin ? false : 5000,
  });

  const flagged = useQuery({
    queryKey: ["service-requests", "live-board"],
    queryFn: () => api.serviceRequests({ ai_flagged: true }),
    refetchInterval: running ? 1200 : isAdmin ? false : 5000,
  });

  const refreshAll = () => {
    for (const key of LIVE_KEYS) queryClient.invalidateQueries({ queryKey: [key] });
  };

  // While the worker runs autonomously nothing else on the page knows a tick
  // landed, so refresh the data views each time the tick counter advances.
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
      if (result && typeof result === "object" && result.error) {
        setNote(`Tick failed: ${result.error}`);
      }
      refreshAll();
    } catch (e) {
      setNote(
        e instanceof ApiError ? e.message : "Replay command failed."
      );
    } finally {
      setBusy(null);
    }
  };

  const data = status.data;
  const percent = data?.percent_released ?? 0;
  const lastTick = data?.worker.last_tick;

  return (
    <Panel className="overflow-hidden">
      {/* ── header ─────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-hair px-5 py-4">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-ink">Live Automation</h2>
            <LiveDot live={running} label={running ? "live" : "idle"} />
          </div>
          <p className="mt-1 max-w-2xl text-xs text-ink-3">
            Replays the archived CWRU recordings as if sensors were streaming them in,
            so the pipeline runs incrementally on arrival.{" "}
            <span className="text-ink-2">
              This is a replay of recorded data — not live telemetry, and no sensors
              are attached.
            </span>
          </p>
        </div>
        {isAdmin && (
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="primary"
              disabled={busy !== null || running || data?.complete}
              onClick={() =>
                run("start", () =>
                  api.replayStart({ interval_seconds: 2, batch_size: 25 })
                )
              }
            >
              {busy === "start" ? "Starting…" : "Start live replay"}
            </Button>
            <Button disabled={busy !== null || !running} onClick={() => run("stop", api.replayStop)}>
              {busy === "stop" ? "Pausing…" : "Pause"}
            </Button>
            <Button
              disabled={busy !== null || running || data?.complete}
              onClick={() => run("step", api.replayStep)}
            >
              {busy === "step" ? "Processing…" : "Step one batch"}
            </Button>
            <Button
              variant="ghost"
              disabled={busy !== null}
              onClick={() => run("reset", api.replayReset)}
            >
              {busy === "reset" ? "Resetting…" : "Reset"}
            </Button>
          </div>
        )}
      </div>

      {/* ── progress + counters ────────────────────────────────────── */}
      <div className="space-y-4 px-5 py-5">
        {isAdmin ? (
          <>
            <div>
              <div className="mb-2 flex items-baseline justify-between gap-3">
                <span className="font-mono text-2xl tabular text-ink">
                  {(data?.released ?? 0).toLocaleString()}
                  <span className="text-ink-3"> / {(data?.total_windows ?? 0).toLocaleString()}</span>
                </span>
                <span className="font-mono text-sm tabular text-ink-2">{percent}%</span>
              </div>
              <ProgressBar fraction={percent / 100} live={running} />
              <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
                <span>windows arrived</span>
                {data?.pending ? <span>· {data.pending} awaiting processing</span> : null}
                {data?.complete && !running && (
                  <span className="text-ok">· archive fully replayed</span>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Metric label="Ticks" value={ticks} accent={running} />
              <Metric
                label="Predicted (last tick)"
                value={lastTick?.predictions_written ?? 0}
              />
              <Metric
                label="Status changes"
                value={lastTick?.equipment_status_changes ?? 0}
                color={
                  lastTick?.equipment_status_changes
                    ? "var(--accent)"
                    : undefined
                }
              />
              <Metric
                label="Requests filed"
                value={lastTick?.service_requests_filed ?? 0}
                color={lastTick?.service_requests_filed ? "var(--accent)" : undefined}
              />
            </div>
          </>
        ) : (
          <p className="rounded-lg border border-dashed border-hair px-4 py-3 text-xs text-ink-3">
            The replay engine is driven by an administrator. The boards below still
            update live while it runs.
          </p>
        )}

        {(note || data?.worker.last_error) && (
          <p className="rounded-md border border-crit/40 bg-crit/10 px-3 py-2 text-xs text-crit">
            {note ?? data?.worker.last_error}
          </p>
        )}
      </div>

      {/* ── the two boards that visibly change ─────────────────────── */}
      <EquipmentStrip query={equipment} />
      <FlaggedFeed query={flagged} />
    </Panel>
  );
}

/* ── equipment: cards flash when a status changes ───────────────────── */

function EquipmentStrip({
  query,
}: {
  query: { data?: Equipment[]; isLoading: boolean; isError: boolean };
}) {
  const previous = useRef<Map<number, string>>(new Map());
  const [changed, setChanged] = useState<Set<number>>(new Set());

  const rows = useMemo(() => query.data ?? [], [query.data]);

  useEffect(() => {
    if (!rows.length) return;
    const justChanged = new Set<number>();
    for (const item of rows) {
      const before = previous.current.get(item.id);
      // `before === undefined` is the first load, not a change -- flashing the
      // whole board on arrival would make the cue meaningless.
      if (before !== undefined && before !== item.status) justChanged.add(item.id);
      previous.current.set(item.id, item.status);
    }
    if (justChanged.size) {
      setChanged(justChanged);
      const timer = setTimeout(() => setChanged(new Set()), 2400);
      return () => clearTimeout(timer);
    }
  }, [rows]);

  return (
    <div className="border-t border-hair px-5 py-4">
      <div className="mb-3 flex items-center gap-2">
        <SectionLabel>Equipment</SectionLabel>
        <span className="text-xs text-ink-3">
          — a card flashes amber the moment the pipeline changes its status
        </span>
      </div>
      {query.isLoading ? (
        <p className="text-sm text-ink-3">Loading equipment…</p>
      ) : query.isError ? (
        <p className="text-sm text-crit">Failed to load equipment.</p>
      ) : rows.length === 0 ? (
        <Empty>No equipment visible to this account.</Empty>
      ) : (
        <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
          {rows.map((item) => (
            <li
              key={item.id}
              className={`fm-transition rounded-lg border border-hair bg-panel-sunken px-3 py-2.5 ${
                changed.has(item.id) ? "fm-flash !border-accent" : ""
              }`}
            >
              <div className="truncate text-sm font-medium text-ink">{item.name}</div>
              <div className="mt-1.5 flex items-center justify-between gap-2">
                <StatusPill status={item.status} />
                <span className="font-mono text-[11px] text-ink-3">
                  {item.equipment_type}
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ── AI-flagged requests: new ones animate in ───────────────────────── */

function FlaggedFeed({
  query,
}: {
  query: { data?: ServiceRequest[]; isLoading: boolean; isError: boolean };
}) {
  const seen = useRef<Set<number> | null>(null);
  const [fresh, setFresh] = useState<Set<number>>(new Set());

  const rows = useMemo(() => query.data ?? [], [query.data]);
  // Keyed on the query having *settled*, not on the list being non-empty.
  // Bailing out while the list was empty meant the baseline was taken from the
  // first response that had anything in it -- so the very first request to be
  // filed, which is exactly the one a reviewer is watching for after pressing
  // Start, was recorded as pre-existing and never animated.
  const loaded = query.data !== undefined;

  useEffect(() => {
    if (!loaded) return;
    const ids = new Set(rows.map((r) => r.id));
    if (seen.current === null) {
      // The first settled response is the baseline, even when it is empty.
      seen.current = ids;
      return;
    }
    // Array.from rather than spread: the project's tsconfig targets a level
    // below es2015 downlevel iteration, so [...set] does not compile.
    const previouslySeen = seen.current;
    const arrived = new Set(
      Array.from(ids).filter((id) => !previouslySeen.has(id))
    );
    seen.current = ids;
    if (arrived.size) {
      setFresh(arrived);
      const timer = setTimeout(() => setFresh(new Set()), 4000);
      return () => clearTimeout(timer);
    }
  }, [rows, loaded]);

  return (
    <div className="border-t border-hair px-5 py-4">
      <div className="mb-3 flex items-center gap-2">
        <SectionLabel>AI-flagged requests</SectionLabel>
        <span className="text-xs text-ink-3">
          — filed automatically when a decision reaches HIGH or CRITICAL
        </span>
      </div>
      {query.isLoading ? (
        <p className="text-sm text-ink-3">Loading requests…</p>
      ) : query.isError ? (
        <p className="text-sm text-crit">Failed to load AI-flagged requests.</p>
      ) : rows.length === 0 ? (
        <Empty>Nothing flagged yet. Start the replay and watch this fill.</Empty>
      ) : (
        <ul className="space-y-1.5">
          {rows.map((request) => {
            const isNew = fresh.has(request.id);
            return (
              <li
                key={request.id}
                className={`fm-transition flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border px-3 py-2 ${
                  isNew
                    ? "fm-rise border-accent bg-accent-wash"
                    : "border-hair bg-panel-sunken"
                }`}
              >
                {isNew && (
                  <span className="rounded bg-accent px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-label text-canvas">
                    new
                  </span>
                )}
                <span className="font-mono text-xs text-ink-3">#{request.id}</span>
                <span className="text-sm text-ink">{request.title}</span>
                {request.predicted_fault && (
                  <span
                    className="font-mono text-xs"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {request.predicted_fault}
                  </span>
                )}
                {request.agent_risk_score !== null && (
                  <span
                    className="font-mono text-xs tabular"
                    style={{
                      color:
                        request.agent_risk_score >= 0.7
                          ? STATUS.critical
                          : "var(--text-secondary)",
                    }}
                  >
                    risk {(request.agent_risk_score * 100).toFixed(0)}%
                  </span>
                )}
                <span className="ml-auto flex items-center gap-2">
                  <StatusPill status={request.priority} />
                  <StatusPill status={request.status} />
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
