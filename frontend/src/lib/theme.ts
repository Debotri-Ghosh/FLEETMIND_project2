/**
 * Chart colours, as literals.
 *
 * Recharts needs real hex strings for SVG `stroke`/`fill` -- it cannot resolve
 * `var(--series-1)` in every prop -- so the validated values are repeated here
 * once, and nowhere else. Keep this file and the `:root` block in globals.css
 * in step; `npm run check:palette` re-runs the validator against both.
 *
 * Produced by the dataviz validator against the real surfaces
 * (--panel #161a21 and --canvas #0e1116), not chosen by eye:
 *   categorical aqua -> blue -> orange -> violet
 *     worst adjacent CVD dE 19.6, normal-vision dE 20.9, all >= 3:1
 *   comparison pair blue/orange: CVD dE 26.8
 *
 * The order is fixed and never cycled: a fifth fault class would fold into
 * "Other" rather than take a generated hue.
 */

export const SURFACE = {
  canvas: "#0e1116",
  panel: "#161a21",
  panelRaised: "#1d222b",
  grid: "#242a33",
  axis: "#39414d",
  border: "#272e39",
} as const;

export const INK = {
  primary: "#e8e6e1",
  secondary: "#aeb4be",
  muted: "#8a919c",
} as const;

/** Amber means live. Never used for a series, a status, or decoration. */
export const ACCENT = "#f2a33c";

/** Fixed categorical order for the four CWRU fault classes. */
export const FAULT_COLORS: Record<string, string> = {
  Normal: "#199e70",
  Ball: "#3987e5",
  InnerRace: "#d95926",
  OuterRace: "#9085e9",
};

/** Fallback for an unexpected label -- deliberately neutral, not a new hue. */
export const FAULT_FALLBACK = INK.muted;

export function faultColor(label: string): string {
  return FAULT_COLORS[label] ?? FAULT_FALLBACK;
}

/** Two-series comparison (local baseline vs federated). */
export const COMPARE = {
  a: "#3987e5",
  b: "#d95926",
} as const;

/** Reserved status colours. Always shipped with a text label, never alone. */
export const STATUS = {
  good: "#0ca30c",
  warn: "#ec835a",
  critical: "#d03b3b",
} as const;

/** Shared Recharts tooltip styling, so every chart's hover layer matches. */
export const TOOLTIP_STYLE = {
  contentStyle: {
    background: SURFACE.panelRaised,
    border: `1px solid ${SURFACE.border}`,
    borderRadius: 8,
    color: INK.primary,
    fontSize: 12,
  },
  labelStyle: { color: INK.secondary },
  itemStyle: { color: INK.primary },
} as const;

export const AXIS_TICK = { fontSize: 11, fill: INK.muted } as const;
