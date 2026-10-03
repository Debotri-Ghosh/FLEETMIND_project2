"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { usePathname, useRouter } from "next/navigation";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";

import { ApiError, api } from "@/lib/api";
import { ROLE_HOME, canOpen, getToken, setToken } from "@/lib/session";
import type { CurrentUser } from "@/lib/types";

interface AuthValue {
  user: CurrentUser | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<CurrentUser>;
  signOut: () => void;
}

const AuthContext = createContext<AuthValue | null>(null);

export function useAuth(): AuthValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside <AuthProvider>");
  return value;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const [token, setTokenState] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  // localStorage does not exist during SSR, so the first client render has to
  // read it in an effect. `hydrated` keeps us from flashing the login screen
  // at a user who is in fact signed in.
  useEffect(() => {
    setTokenState(getToken());
    setHydrated(true);
    const sync = () => setTokenState(getToken());
    window.addEventListener("fleetmind:session", sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener("fleetmind:session", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  // The token is only a claim; /auth/me is what makes it a session. A token
  // that is expired, revoked or for a deactivated account fails here, which
  // is why we never read the role out of the JWT on the client.
  const me = useQuery({
    queryKey: ["auth", "me", token],
    queryFn: () => api.me(),
    enabled: hydrated && Boolean(token),
    retry: false,
    staleTime: 5 * 60_000,
  });

  useEffect(() => {
    if (me.isError && me.error instanceof ApiError && me.error.status === 401) {
      setToken(null);
    }
  }, [me.isError, me.error]);

  const signIn = useCallback(
    async (email: string, password: string) => {
      const result = await api.login(email, password);
      setToken(result.access_token);
      setTokenState(result.access_token);
      queryClient.clear(); // never show the previous user's cached rows
      return result.user;
    },
    [queryClient]
  );

  const signOut = useCallback(() => {
    setToken(null);
    setTokenState(null);
    queryClient.clear();
  }, [queryClient]);

  const value: AuthValue = {
    user: token && me.data ? me.data : null,
    loading: !hydrated || (Boolean(token) && me.isLoading),
    signIn,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/**
 * Redirects a signed-out or wrongly-roled visitor away from a page.
 *
 * This is a usability guard, not a security one. It stops someone landing on
 * a page full of "Failed to load" panels; it is not what keeps them out of
 * the data. Every endpoint behind these pages enforces the same rule itself,
 * and `backend/tests/test_auth.py` proves it by calling the API directly with
 * each role. Deleting this file would make the app ugly, not insecure.
 */
export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (loading) return;
    if (!user) {
      router.replace(`/login?next=${encodeURIComponent(pathname)}`);
    } else if (!canOpen(user, pathname)) {
      router.replace(ROLE_HOME[user.role]);
    }
  }, [user, loading, pathname, router]);

  if (loading) {
    return (
      <main className="mx-auto max-w-6xl px-8 py-10">
        <p className="text-ink-3">Checking your session…</p>
      </main>
    );
  }
  if (!user || !canOpen(user, pathname)) {
    return (
      <main className="mx-auto max-w-6xl px-8 py-10">
        <p className="text-ink-3">Redirecting…</p>
      </main>
    );
  }
  return <>{children}</>;
}
