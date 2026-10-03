"use client";

/**
 * The Frequency Lab.
 *
 * The misconception this exists to kill: that the frequency view is a second,
 * separately collected measurement. It is not — it is the same stored samples,
 * re-expressed. So the interaction is deliberately staged:
 *
 *   1. the waveform is on screen first, by itself;
 *   2. the user presses a button that says what it is about to do;
 *   3. the chart morphs — the same panel, the same box — into the spectrum.
 *
 * The morph matters. If the spectrum simply appeared in a second panel the
 * page would read as "here is another thing we measured". Transforming the
 * chart that is already there, after an explicit press, reads as "that signal,
 * seen differently", which is the true statement.
 *
 * Both halves come from one API response (`/signal-window/{id}/spectrum`), so
 * there is no second fetch at the moment of transform that could suggest new
 * data arrived.
 */

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api } from "@/lib/api";
import { ACCENT, AXIS_TICK, INK, TOOLTIP_STYLE, faultColor } from "@/lib/theme";
import { ErrorRetry } from "./components";
import { Button, Metric, SectionLabel } from "./ui";

type Domain = "time" | "frequency";

export function FrequencyLab({ windowId }: { windowId: number }) {
  const [domain, setDomain] = useState<Domain>("time");
  const [morphing, setMorphing] = useState(false);

  const query = useQuery({
    queryKey: ["window-spectrum", windowId],
    queryFn: () => api.windowSpectrum(windowId),
  });

  // A new window always starts in the time domain: the point of the lab is the
  // transform, and landing already-transformed skips it.
  useEffect(() => {
    setDomain("time");
    setMorphing(false);
  }, [windowId]);

  const data = query.data;

  const waveform = useMemo(
    () =>
      (data?.raw_values ?? []).map((v, i) => ({
        x: Number((((i / (data?.sampling_rate_hz || 1)) * 1000)).toFixed(3)),
        y: v,
      })),
    [data]
  );

  const spectrum = useMemo(
    () =>
      (data?.spectrum_freqs_hz ?? []).map((hz, i) => ({
        x: hz,
        y: data?.spectrum_magnitudes[i] ?? 0,
      })),
    [data]
  );

  if (query.isLoading)
    return <p className="text-sm text-ink-3">Loading signal…</p>;
  if (query.isError || !data)
    return (
      <ErrorRetry
        message="Failed to load this signal window."
        onRetry={() => query.refetch()}
      />
    );

  const band = data.matched_frequency;
  const isFrequency = domain === "frequency";
  const series = isFrequency ? spectrum : waveform;
  const color = faultColor(data.label);

  const transform = () => {
    // The 520ms pause is the animation, not a fake loading delay: the chart
    // fades and rescales, then the other domain fades in. Without it the swap
    // is instant and reads as a tab change rather than a transformation.
    setMorphing(true);
    setTimeout(() => {
      setDomain((d) => (d === "time" ? "frequency" : "time"));
      setMorphing(false);
    }, 520);
  };

  return (
    <div className="space-y-4">
      {/* ── what you are looking at ──────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
        <span className="flex items-center gap-2">
          <SectionLabel>Window</SectionLabel>
          <span className="font-mono text-ink">#{data.window_id}</span>
        </span>
        <span className="flex items-center gap-2">
          <SectionLabel>Documented class</SectionLabel>
          <span className="inline-flex items-center gap-1.5 font-medium" style={{ color }}>
            <span
              className="h-2 w-2 rounded-full"
              style={{ background: color }}
              aria-hidden="true"
            />
            {data.label}
          </span>
        </span>
        <span className="flex items-center gap-2">
          <SectionLabel>Samples</SectionLabel>
          <span className="font-mono tabular text-ink">
            {data.n_samples.toLocaleString()} @ {data.sampling_rate_hz.toLocaleString()} Hz
          </span>
        </span>
        <span className="flex items-center gap-2">
          <SectionLabel>Duration</SectionLabel>
          <span className="font-mono tabular text-ink">
            {(data.duration_seconds * 1000).toFixed(1)} ms
          </span>
        </span>
      </div>

      {/* ── the transform control ────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-hair bg-panel-sunken px-4 py-3">
        <Button variant="primary" onClick={transform} disabled={morphing}>
          {morphing
            ? "Transforming…"
            : isFrequency
              ? "◀ Back to time domain"
              : "Apply Fourier Transform ▶"}
        </Button>
        <p className="min-w-[16rem] flex-1 text-xs text-ink-2">
          Frequency content is <span className="font-medium text-ink">computed from this
          exact recording</span> via Fourier Transform — nothing new is collected, and no
          second measurement is taken. The spectrum below is the waveform above, re-expressed.
        </p>
      </div>

      {/* ── the chart that morphs ────────────────────────────────── */}
      <div
        className="rounded-lg border border-hair bg-panel-sunken p-3 transition-all duration-500"
        style={{
          opacity: morphing ? 0.25 : 1,
          transform: morphing ? "scaleY(0.92)" : "none",
        }}
      >
        <div className="mb-2 flex items-baseline justify-between">
          <SectionLabel>
            {isFrequency ? "Frequency domain — magnitude spectrum" : "Time domain — raw vibration"}
          </SectionLabel>
          <span className="font-mono text-[11px] text-ink-3">
            {isFrequency ? "magnitude vs Hz" : "amplitude (g) vs ms"}
          </span>
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={series} margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="x"
              type="number"
              domain={[0, "dataMax"]}
              tick={AXIS_TICK}
              tickLine={false}
              label={{
                value: isFrequency ? "Hz" : "ms",
                position: "insideBottomRight",
                offset: -2,
                fontSize: 11,
                fill: INK.muted,
              }}
            />
            <YAxis tick={AXIS_TICK} tickLine={false} width={48} />
            <Tooltip
              {...TOOLTIP_STYLE}
              formatter={(v: number) => v.toFixed(4)}
              labelFormatter={(x) => (isFrequency ? `${x} Hz` : `${x} ms`)}
            />
            {/* The defect band only means anything in the frequency domain. */}
            {isFrequency && band && (
              <ReferenceArea
                x1={band.band_low_hz}
                x2={band.band_high_hz}
                fill={ACCENT}
                fillOpacity={0.16}
                stroke={ACCENT}
                strokeOpacity={0.4}
              />
            )}
            {isFrequency && band && (
              <ReferenceLine
                x={band.hz}
                stroke={ACCENT}
                strokeDasharray="4 3"
                label={{
                  value: band.name,
                  position: "top",
                  fontSize: 11,
                  fill: ACCENT,
                }}
              />
            )}
            <Line
              dataKey="y"
              stroke={color}
              dot={false}
              strokeWidth={isFrequency ? 1.5 : 1}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── what the spectrum shows ──────────────────────────────── */}
      {isFrequency && (
        <div className="fm-rise space-y-3">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Metric
              label="Defect band"
              value={band ? `${band.hz} Hz` : "—"}
              sub={
                band
                  ? `${band.name} · ${band.band_low_hz}–${band.band_high_hz} Hz`
                  : `${data.label} has no characteristic fault frequency`
              }
            />
            <Metric
              label="Energy in band"
              value={
                data.band_energy_fraction === null
                  ? "—"
                  : data.band_energy_fraction.toFixed(6)
              }
              sub="share of total spectral energy"
            />
            <Metric
              label="Overall peak"
              value={data.overall_peak_hz !== null ? `${data.overall_peak_hz} Hz` : "—"}
              sub="where the energy actually is"
            />
          </div>

          <p className="text-xs text-ink-3">
            The shaded band is bearing geometry — the frequency a {data.label} defect
            produces at this shaft speed. Energy sitting on it is independent physical
            corroboration.{" "}
            <span className="text-ink-2">
              It is <span className="font-medium text-ink">not</span> what the model looked
              at: the CNN consumes a log-scaled, normalised spectrum and was never given
              these frequencies.
            </span>
          </p>

          {band &&
            data.band_energy_fraction !== null &&
            data.overall_peak_hz !== null &&
            (data.overall_peak_hz < band.band_low_hz ||
              data.overall_peak_hz > band.band_high_hz) && (
              <p className="rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
                Worth saying out loud: on this window the dominant energy is at{" "}
                <span className="font-mono">{data.overall_peak_hz} Hz</span>, outside the{" "}
                {band.name} band. The physics does not corroborate the label here — this
                view shows that rather than hiding it.
              </p>
            )}

          <p className="text-xs text-ink-3">
            All defect frequencies at this shaft speed:{" "}
            <span className="font-mono text-ink-2">
              {Object.entries(data.all_characteristic_frequencies)
                .map(([k, v]) => `${k} ${v} Hz`)
                .join("  ·  ")}
            </span>
          </p>
        </div>
      )}
    </div>
  );
}
