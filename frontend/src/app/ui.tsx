"use client";

/**
 * Shared presentation primitives for the instrument-panel theme.
 *
 * Kept apart from components.tsx (which holds the domain components) so the
 * visual language is defined once. Anything here is purely presentational and
 * holds no data-fetching logic.
 */

import { useEffect, useRef, useState } from "react";

import { STATUS } from "@/lib/theme";

export function Panel({
  children,
  className = "",
  as: Tag = "section",
}: {
  children: React.ReactNode;
  className?: string;
  as?: "section" | "div" | "article";
}) {
  return (
    <Tag
      className={`rounded-panel border border-hair bg-panel shadow-panel ${className}`}
    >
      {children}
    </Tag>
  );
}

/** The small uppercase label that carries most of the chrome in this theme. */
export function SectionLabel({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={`text-[11px] font-semibold uppercase tracking-label text-ink-3 ${className}`}
    >
      {children}
    </span>
  );
}

export function PanelHeader({
  title,
  hint,
  right,
}: {
  title: string;
  hint?: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-hair px-5 py-4">
      <div className="min-w-0">
        <h2 className="text-base font-semibold text-ink">{title}</h2>
        {hint && <p className="mt-1 max-w-3xl text-xs text-ink-3">{hint}</p>}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  );
}

/**
 * The live indicator. Amber + pulsing means the worker is running; grey and
 * still means it is not. This is the single most important piece of state on
 * the dashboard, so it is also spelled out in words beside the dot -- colour
 * and motion are both redundant cues, not the only ones.
 */
export function LiveDot({ live, label }: { live: boolean; label?: string }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span className="relative inline-flex h-2.5 w-2.5">
        {live && (
          <span
            className="fm-ring absolute inset-0 rounded-full"
            aria-hidden="true"
          />
        )}
        <span
          className={`fm-transition relative inline-flex h-2.5 w-2.5 rounded-full ${
            live ? "fm-pulse bg-accent" : "bg-hair-strong"
          }`}
        />
      </span>
      {label !== undefined && (
        <SectionLabel className={live ? "!text-accent" : undefined}>
          {label}
        </SectionLabel>
      )}
    </span>
  );
}

const STATUS_STYLE: Record<string, { color: string; label: string }> = {
  active: { color: STATUS.good, label: "active" },
  available: { color: STATUS.good, label: "available" },
  completed: { color: STATUS.good, label: "completed" },
  resolved: { color: STATUS.good, label: "resolved" },
  maintenance: { color: STATUS.warn, label: "maintenance" },
  busy: { color: STATUS.warn, label: "busy" },
  reported: { color: STATUS.warn, label: "reported" },
  dispatched: { color: STATUS.warn, label: "dispatched" },
  in_progress: { color: STATUS.warn, label: "in progress" },
  traveling: { color: STATUS.warn, label: "traveling" },
  assigned: { color: STATUS.warn, label: "assigned" },
  retired: { color: "#8a919c", label: "retired" },
  offline: { color: "#8a919c", label: "offline" },
  cancelled: { color: "#8a919c", label: "cancelled" },
  critical: { color: STATUS.critical, label: "critical" },
  high: { color: STATUS.warn, label: "high" },
  medium: { color: "#3987e5", label: "medium" },
  low: { color: "#8a919c", label: "low" },
};

/**
 * A status never relies on colour alone: the pill always carries a dot *and*
 * the word. That is the mitigation the dataviz rules require for status hues,
 * which sit close to some series hues by design.
 */
export function StatusPill({ status }: { status: string }) {
  const spec = STATUS_STYLE[status] ?? { color: "#8a919c", label: status };
  return (
    <span
      className="fm-transition inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium"
      style={{
        color: spec.color,
        borderColor: `${spec.color}44`,
        background: `${spec.color}14`,
      }}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: spec.color }}
        aria-hidden="true"
      />
      {spec.label}
    </span>
  );
}

/**
 * A number that visibly changes.
 *
 * When the value increases it briefly takes the live accent, so a reviewer
 * watching the screen sees the counter move rather than having to compare it
 * with a remembered previous value. The flash is keyed on the value, so React
 * restarts the animation on every change.
 */
export function Ticker({
  value,
  className = "",
}: {
  value: number;
  className?: string;
}) {
  const previous = useRef(value);
  const [bumped, setBumped] = useState(false);

  useEffect(() => {
    if (value !== previous.current) {
      previous.current = value;
      setBumped(true);
      const timer = setTimeout(() => setBumped(false), 900);
      return () => clearTimeout(timer);
    }
  }, [value]);

  return (
    <span
      className={`fm-transition font-mono tabular ${
        bumped ? "text-accent" : "text-ink"
      } ${className}`}
    >
      {value.toLocaleString()}
    </span>
  );
}

export function Metric({
  label,
  value,
  sub,
  accent = false,
  color,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  accent?: boolean;
  color?: string;
}) {
  return (
    <div className="rounded-lg border border-hair bg-panel-sunken px-4 py-3">
      <SectionLabel>{label}</SectionLabel>
      <div
        className="mt-1 font-mono text-2xl tabular"
        style={{ color: color ?? (accent ? "var(--accent)" : "var(--text)") }}
      >
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-ink-3">{sub}</div>}
    </div>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "default",
  type = "button",
  title,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "default" | "primary" | "ghost";
  type?: "button" | "submit";
  title?: string;
}) {
  const base =
    "fm-transition inline-flex items-center justify-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-40";
  const styles = {
    primary: "bg-accent text-canvas hover:brightness-110",
    default:
      "border border-hair-strong bg-panel-raised text-ink hover:border-ink-3 hover:bg-hair",
    ghost: "text-ink-2 hover:bg-panel-raised hover:text-ink",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`${base} ${styles}`}
    >
      {children}
    </button>
  );
}

/** A labelled progress bar. The sweep only runs while `live` is true. */
export function ProgressBar({
  fraction,
  live,
}: {
  fraction: number;
  live: boolean;
}) {
  const pct = Math.max(0, Math.min(1, fraction)) * 100;
  return (
    <div
      className="relative h-2 w-full overflow-hidden rounded-full bg-panel-sunken"
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className={`h-full rounded-full transition-[width] duration-700 ease-out ${
          live ? "bg-accent" : "bg-hair-strong"
        }`}
        style={{ width: `${pct}%` }}
      />
      {live && pct < 100 && (
        <span className="fm-sweep absolute inset-0 overflow-hidden" aria-hidden="true" />
      )}
    </div>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return (
    <p className="rounded-lg border border-dashed border-hair px-4 py-6 text-center text-sm text-ink-3">
      {children}
    </p>
  );
}

export function Th({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <th
      className={`border-b border-hair px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-label text-ink-3 ${className}`}
    >
      {children}
    </th>
  );
}

export function Td({
  children,
  className = "",
  colSpan,
}: {
  children: React.ReactNode;
  className?: string;
  colSpan?: number;
}) {
  return (
    <td
      colSpan={colSpan}
      className={`border-b border-hair/60 px-3 py-2.5 align-top text-sm ${className}`}
    >
      {children}
    </td>
  );
}
