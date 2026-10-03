/** @type {import('tailwindcss').Config} */
// Tokens live in globals.css as CSS custom properties; this file only exposes
// them to Tailwind's utility classes. Keeping one source means a colour can be
// changed in one place and the charts (which read the variables directly via
// src/lib/theme.ts) stay in step with the markup.
module.exports = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "var(--canvas)",
        panel: "var(--panel)",
        "panel-raised": "var(--panel-raised)",
        "panel-sunken": "var(--panel-sunken)",
        hair: "var(--border)",
        "hair-strong": "var(--border-strong)",
        ink: "var(--text)",
        "ink-2": "var(--text-secondary)",
        "ink-3": "var(--text-muted)",
        accent: "var(--accent)",
        "accent-dim": "var(--accent-dim)",
        "accent-wash": "var(--accent-wash)",
        ok: "var(--status-good)",
        warn: "var(--status-warn)",
        crit: "var(--status-critical)",
        "s-1": "var(--series-1)",
        "s-2": "var(--series-2)",
        "s-3": "var(--series-3)",
        "s-4": "var(--series-4)",
      },
      fontFamily: {
        sans: ["var(--font-plex-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-plex-mono)", "ui-monospace", "monospace"],
      },
      borderRadius: {
        DEFAULT: "var(--radius)",
        panel: "var(--radius)",
      },
      boxShadow: {
        panel: "var(--shadow-panel)",
      },
      letterSpacing: {
        // For the small uppercase labels that carry most of the chrome.
        label: "0.08em",
      },
    },
  },
  plugins: [],
};
