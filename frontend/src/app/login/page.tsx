"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { useAuth } from "../auth-provider";
import { ApiError } from "@/lib/api";
import { ROLE_HOME, canOpen } from "@/lib/session";

const DEMO_LOGINS = [
  {
    label: "Admin — sees everything",
    email: "admin@fleetmind.example.com",
    password: "admin-demo-2026",
  },
  {
    label: "Operator — Plant A (0 HP) only",
    email: "operator.a@fleetmind.example.com",
    password: "operator-demo-2026",
  },
  {
    label: "Operator — Plant C (2 HP) only",
    email: "operator.c@fleetmind.example.com",
    password: "operator-demo-2026",
  },
  {
    label: "Customer — Acme Manufacturing",
    email: "customer.acme@fleetmind.example.com",
    password: "customer-demo-2026",
  },
  {
    label: "Customer — Northwind Steel",
    email: "customer.northwind@fleetmind.example.com",
    password: "customer-demo-2026",
  },
];

function LoginForm() {
  const { user, signIn } = useAuth();
  const router = useRouter();
  const params = useSearchParams();
  const next = params.get("next");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Already signed in (e.g. hit /login directly) — go where they belong.
  useEffect(() => {
    if (user) router.replace(canOpen(user, next ?? "") ? (next ?? ROLE_HOME[user.role]) : ROLE_HOME[user.role]);
  }, [user, next, router]);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const signedIn = await signIn(email.trim(), password);
      const target =
        next && canOpen(signedIn, next) ? next : ROLE_HOME[signedIn.role];
      router.replace(target);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Could not sign in. Is the API running?"
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto flex max-w-md flex-col gap-6 px-6 py-16">
      <div>
        <h1 className="text-2xl font-bold">FleetMind</h1>
        <p className="text-sm text-ink-2">
          Sign in to continue. What you can see depends on your role, and the
          restriction is enforced by the API — not by hiding buttons.
        </p>
      </div>

      <form onSubmit={onSubmit} className="space-y-3 rounded-lg border border-hair bg-panel p-5">
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Email</span>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded border border-hair-strong px-3 py-2"
            autoComplete="username"
          />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Password</span>
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded border border-hair-strong px-3 py-2"
            autoComplete="current-password"
          />
        </label>

        {error && (
          <p role="alert" className="rounded border border-crit/40 bg-crit/10 p-2 text-sm text-crit">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="w-full rounded bg-accent px-3 py-2 font-medium text-canvas hover:brightness-110 disabled:opacity-50"
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <div className="rounded-lg border border-hair bg-panel-sunken p-4 text-sm">
        <p className="mb-2 font-medium">Demo logins</p>
        <p className="mb-3 text-xs text-ink-2">
          These are seeded demo accounts with published passwords, so a reviewer
          can sign in as each role and check that the restrictions are real.
          The two operators sit at <em>different</em> sites on purpose.
        </p>
        <ul className="space-y-1">
          {DEMO_LOGINS.map((demo) => (
            <li key={demo.email}>
              <button
                type="button"
                onClick={() => {
                  setEmail(demo.email);
                  setPassword(demo.password);
                }}
                className="text-left text-accent hover:underline"
              >
                {demo.label}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </main>
  );
}

export default function LoginPage() {
  // useSearchParams needs a Suspense boundary for the static export.
  return (
    <Suspense fallback={<main className="px-8 py-16 text-ink-3">Loading…</main>}>
      <LoginForm />
    </Suspense>
  );
}
