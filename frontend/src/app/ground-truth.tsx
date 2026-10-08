"use client";

/**
 * "Documented by the dataset creators" vs "predicted by our model".
 *
 * The question this answers is the one a reviewer asks first: *how do you know
 * the model isn't just reading the label?* Showing two numbers in a table does
 * not answer it — the two have to look like they come from different places,
 * because they do.
 *
 * So: two panels, side by side, visually unlike each other. The left is inert
 * — hairline border, muted ink, a dataset citation — and reads as an archival
 * record. The right carries the fault class's own series colour and the
 * model's confidence, and reads as a live output. The verdict strip between
 * them states agreement or disagreement in words, and is as prominent when the
 * model is **wrong** as when it is right; a panel that only celebrates matches
 * is evidence of nothing.
 */

import { faultColor } from "@/lib/theme";
import type { EvidencePackage } from "@/lib/types";
import { SectionLabel } from "./ui";

function Field({
  label,
  value,
  mono = true,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-hair/50 py-1.5 last:border-0">
      <span className="text-xs text-ink-3">{label}</span>
      <span className={`text-sm text-ink ${mono ? "font-mono tabular" : ""}`}>{value}</span>
    </div>
  );
}

export function GroundTruthVsPrediction({ evidence }: { evidence: EvidencePackage }) {
  const agrees = evidence.prediction_matches_documented;
  const predictedColor = faultColor(evidence.predicted_label.split(" ")[0]);
  const documentedColor = faultColor(evidence.documented_fault_type);

  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-2">
        {/* ── archival record ─────────────────────────────────────── */}
        <div className="rounded-lg border border-dashed border-hair-strong bg-panel-sunken p-4">
          <div className="mb-3 flex items-center gap-2">
            <span
              className="grid h-6 w-6 place-items-center rounded border border-hair-strong text-[11px] text-ink-3"
              aria-hidden="true"
            >
              ❏
            </span>
            <div>
              <SectionLabel>Documented by dataset creators</SectionLabel>
              <p className="text-[11px] text-ink-3">
                Machined into the bearing before the rig was run. Ground truth.
              </p>
            </div>
          </div>

          <div
            className="mb-3 flex items-baseline gap-2 text-2xl font-semibold"
            style={{ color: documentedColor }}
          >
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: documentedColor }}
              aria-hidden="true"
            />
            {evidence.documented_fault_type}
          </div>

          <Field
            label="Defect diameter"
            value={
              evidence.documented_fault_diameter_in !== null
                ? `${evidence.documented_fault_diameter_in}″`
                : "—"
            }
          />
          <Field label="Sensor position" value={evidence.sensor_position} />
          <Field label="Motor load" value={`${evidence.site_load_hp} HP`} />
          <Field
            label="Nominal speed"
            value={evidence.nominal_rpm !== null ? `${evidence.nominal_rpm} rpm` : "—"}
          />
          <Field label="Window label" value={evidence.documented_window_label} />
          <Field
            label="Source file"
            value={<span className="text-xs">{evidence.source_file}</span>}
          />

          <p className="mt-3 text-[11px] leading-relaxed text-ink-3">
            Case Western Reserve University Bearing Data Center —
            engineering.case.edu/bearingdatacenter
          </p>
        </div>

        {/* ── model output ────────────────────────────────────────── */}
        <div
          className="rounded-lg border bg-panel-raised p-4"
          style={{ borderColor: `${predictedColor}66` }}
        >
          <div className="mb-3 flex items-center gap-2">
            <span
              className="grid h-6 w-6 place-items-center rounded text-[11px]"
              style={{ background: `${predictedColor}22`, color: predictedColor }}
              aria-hidden="true"
            >
              ◆
            </span>
            <div>
              <SectionLabel>Predicted by our model, blind to the label</SectionLabel>
              <p className="text-[11px] text-ink-3">
                1D CNN over the window&apos;s FFT magnitude. Never shown the row above.
              </p>
            </div>
          </div>

          <div
            className="mb-3 flex items-baseline gap-2 text-2xl font-semibold"
            style={{ color: predictedColor }}
          >
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: predictedColor }}
              aria-hidden="true"
            />
            {evidence.predicted_label}
          </div>

          <Field
            label="Confidence"
            value={
              evidence.model_confidence !== null
                ? `${(evidence.model_confidence * 100).toFixed(2)}%`
                : "—"
            }
          />
          <Field label="Checkpoint" value={<span className="text-xs">{evidence.model_version ?? "—"}</span>} />
          <Field label="Signal RMS" value={evidence.signal_rms.toFixed(6)} />
          <Field label="Peak-to-peak" value={evidence.signal_peak.toFixed(6)} />
          <Field
            label="Diagnosis confidence"
            value={`${(evidence.diagnosis_confidence * 100).toFixed(1)}%`}
          />
          <Field
            label={
              evidence.decided_by === "expected_cost_policy"
                ? "P(faulted)"
                : "Risk score (heuristic)"
            }
            value={
              evidence.risk_score !== null
                ? `${(evidence.risk_score * 100).toFixed(1)}%`
                : "—"
            }
          />

          <p
            className="mt-3 text-[11px] leading-relaxed text-ink-3"
            data-testid="split-note"
          >
            {evidence.recording_in_training_split
              ? "This recording was in the site model's training split, so agreement here is in-sample — it is not evidence the model generalises."
              : "This recording was held out of the site model's training split (experiment-level), so no window from it was trained on."}
          </p>
        </div>
      </div>

      {/* ── the verdict ─────────────────────────────────────────── */}
      <div
        className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border px-4 py-3 text-sm"
        style={{
          borderColor: agrees ? "var(--status-good)66" : "var(--status-warn)66",
          background: agrees ? "rgba(12,163,12,0.10)" : "rgba(236,131,90,0.10)",
        }}
      >
        <span
          className="font-semibold"
          style={{ color: agrees ? "var(--status-good)" : "var(--status-warn)" }}
        >
          {agrees ? "Agreement" : "Disagreement"}
        </span>
        <span className="text-ink-2">
          {agrees ? (
            <>
              The model independently reached{" "}
              <span className="font-medium text-ink">{evidence.predicted_label}</span>, which
              matches the documented fault.
            </>
          ) : (
            <>
              The model said{" "}
              <span className="font-medium text-ink">{evidence.predicted_label}</span> where
              the dataset documents{" "}
              <span className="font-medium text-ink">{evidence.documented_fault_type}</span>.
              This is a miss, shown rather than filtered out.
            </>
          )}
        </span>
      </div>
    </div>
  );
}
