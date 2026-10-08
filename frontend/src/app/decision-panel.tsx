"use client";

/**
 * What was decided for this machine, and why — in the decision layer's own terms.
 *
 * The expected-cost policy is the authoritative path: it reads the classifier's
 * full probability distribution, prices every action under it, and picks the
 * cheapest. A machine is flagged only when a sequential test over its windows
 * crosses its boundary. The heuristic baseline is shown beside it for
 * comparison and labelled as such, never as the decision.
 */

import { faultColor } from "@/lib/theme";
import type { EvidencePackage } from "@/lib/types";
import { SectionLabel } from "./ui";

const ACTION_LABEL: Record<string, string> = {
  monitor: "Monitor",
  inspect: "Inspect",
  schedule_maintenance: "Schedule maintenance",
  stop_machine: "Stop machine",
};

const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

export function DecisionPanel({ evidence: e }: { evidence: EvidencePackage }) {
  const pa = e.policy_assessment;

  if (!pa) {
    return (
      <div className="rounded-lg border border-hair bg-panel p-4 text-sm">
        <p className="text-ink-2">
          This request was filed by the <strong>heuristic baseline</strong>
          {e.decided_by ? ` (${e.decided_by})` : ""}: no class distribution is stored for its
          prediction, so the expected-cost policy cannot assess it.
        </p>
        <p className="mt-1 font-mono tabular text-xs text-ink-3">
          Baseline urgency {e.baseline_urgency} · risk {e.baseline_risk_score.toFixed(4)}
        </p>
      </div>
    );
  }

  const costs = Object.entries(pa.expected_costs).sort((a, b) => a[1] - b[1]);
  const maxCost = Math.max(...costs.map(([, c]) => c), 1);
  const sprt = e.sprt;

  return (
    <div className="space-y-4 rounded-lg border border-hair bg-panel p-4" data-testid="decision-panel">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <p className="text-xs text-ink-3">Decided by the expected-cost policy</p>
          <p className="text-lg font-semibold text-ink">{ACTION_LABEL[pa.action] ?? pa.action}</p>
          <p className="text-xs text-ink-2">{pa.action_text}</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-ink-3">P(faulted) = 1 − P(Normal)</p>
          <p className="font-mono tabular text-lg text-ink">{pct(pa.p_faulted, 2)}</p>
          {pa.flag_threshold_p_faulted !== null && (
            <p className="text-[11px] text-ink-3">
              one window alone favours repair at ≥ {pct(pa.flag_threshold_p_faulted, 0)}
            </p>
          )}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <SectionLabel>Expected cost of each action</SectionLabel>
          <p className="mb-2 text-[11px] text-ink-3">
            Relative units: an unplanned run-to-failure costs 100. Lowest wins.
          </p>
          <ul className="space-y-1.5" data-testid="expected-costs">
            {costs.map(([action, cost]) => {
              const chosen = action === pa.action;
              return (
                <li key={action} className="text-xs">
                  <div className="flex justify-between">
                    <span className={chosen ? "font-semibold text-ink" : "text-ink-2"}>
                      {ACTION_LABEL[action] ?? action}
                      {chosen ? " — chosen" : ""}
                      {pa.never_optimal_actions.includes(action) ? " (never optimal)" : ""}
                    </span>
                    <span className="font-mono tabular text-ink">{cost.toFixed(2)}</span>
                  </div>
                  <div className="mt-0.5 h-1.5 rounded bg-hair">
                    <div
                      className={`h-1.5 rounded ${chosen ? "bg-ink" : "bg-ink-3"}`}
                      style={{ width: `${(cost / maxCost) * 100}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
          {pa.never_optimal_actions.length > 0 && (
            <p className="mt-2 text-[11px] text-ink-3">
              Stopping now is never cheaper than a planned repair: pricing it needs a
              degradation rate, and the CWRU archive has no run-to-failure data to supply one.
            </p>
          )}
        </div>

        <div>
          <SectionLabel>Classifier probability, all four classes</SectionLabel>
          <ul className="mt-2 space-y-1.5" data-testid="class-probabilities">
            {Object.entries(pa.class_probabilities).map(([label, p]) => (
              <li key={label} className="text-xs">
                <div className="flex justify-between">
                  <span className="text-ink-2">{label}</span>
                  <span className="font-mono tabular text-ink">{pct(p)}</span>
                </div>
                <div className="mt-0.5 h-1.5 rounded bg-hair">
                  <div
                    className="h-1.5 rounded"
                    style={{ width: `${p * 100}%`, background: faultColor(label) }}
                  />
                </div>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[11px] text-ink-3">
            Uncalibrated: on held-out recordings this model is over-confident (ECE 0.2146).
            Temperature scaling was tested and made it worse, so it is not applied.
          </p>
        </div>
      </div>

      {sprt && (
        <div className="rounded border border-hair/60 bg-panel-sunken p-3 text-xs" data-testid="sprt">
          <SectionLabel>Why the machine was flagged — sequential test</SectionLabel>
          <p className="mt-1 text-ink-2">
            {sprt.flagged ? (
              <>
                Flagged after <strong className="font-mono tabular">{sprt.windows_examined}</strong> of{" "}
                {sprt.windows_available} window(s): accumulated evidence reached{" "}
                <span className="font-mono tabular">{sprt.max_evidence.toFixed(2)}</span> against the
                boundary <span className="font-mono tabular">{sprt.upper_boundary.toFixed(2)}</span>.
              </>
            ) : (
              <>
                Not flagged over {sprt.windows_available} window(s); evidence peaked at{" "}
                <span className="font-mono tabular">{sprt.max_evidence.toFixed(2)}</span>, below the
                boundary <span className="font-mono tabular">{sprt.upper_boundary.toFixed(2)}</span>.
              </>
            )}{" "}
            Targets: false alarm α = {sprt.alpha}, miss β = {sprt.beta}. Consecutive windows are
            correlated, so these are design targets, not guaranteed rates.
          </p>
        </div>
      )}

      <p className="border-t border-hair/60 pt-2 text-xs text-ink-3" data-testid="baseline">
        Heuristic baseline on this window, for comparison only: urgency{" "}
        <span className="font-mono">{e.baseline_urgency}</span>, risk{" "}
        <span className="font-mono tabular">{e.baseline_risk_score.toFixed(4)}</span>. Built from
        hand-set constants; retained as a measured baseline, not used to decide.
      </p>
    </div>
  );
}
