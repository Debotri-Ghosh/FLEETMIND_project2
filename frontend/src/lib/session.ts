"use client";

/**
 * Client-side session handling.
 *
 * The token is kept in localStorage and sent as an Authorization header.
 *
 * The honest tradeoff, stated plainly because it matters: an httpOnly cookie
 * would be safer against XSS, but the API is a separate origin
 * (localhost:8000 vs localhost:3000), so a cookie would need
 * SameSite=None + Secure + credentialed CORS and a TLS terminator in front of
 * both. For a thesis demo that is more moving parts than the threat warrants,
 * and localStorage does not weaken the thing this slice is actually
 * demonstrating: **every restriction is enforced by the API**, so a stolen or
 * hand-edited token still cannot read another customer's data or another
 * site's queue. Swapping to cookies would be a deployment change, not a
 * redesign. Noted as a limitation in the README.
 */

import type { CurrentUser, Role } from "./types";

const TOKEN_KEY = "fleetmind.token";

/** localStorage throws in some privacy modes; a session is never worth a crash. */
function safeGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key: string, value: string | null) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    /* ignore: the session just won't survive a reload */
  }
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return safeGet(TOKEN_KEY);
}

export function setToken(token: string | null) {
  if (typeof window === "undefined") return;
  safeSet(TOKEN_KEY, token);
  // Other tabs listen for this so signing out of one signs out of all.
  window.dispatchEvent(new Event("fleetmind:session"));
}

/** What each role is allowed to open. Mirrors the API, never replaces it. */
export const ROLE_HOME: Record<Role, string> = {
  customer: "/customer",
  operator: "/operator",
  admin: "/",
};

export function canOpen(user: CurrentUser | null, pathname: string): boolean {
  if (!user) return false;
  if (user.role === "admin") return true;
  if (pathname.startsWith("/operator")) return user.role === "operator";
  if (pathname.startsWith("/customer")) return user.role === "customer";
  return true; // the dashboard is readable by any signed-in role
}
