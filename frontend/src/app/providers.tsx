"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

import { AuthProvider } from "./auth-provider";

export default function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 30_000 },
        },
      })
  );
  return (
    <QueryClientProvider client={queryClient}>
      {/* AuthProvider sits inside QueryClientProvider: it uses a query for
          /auth/me and clears the cache on sign-in/out, so one user's rows can
          never be served to the next. */}
      <AuthProvider>{children}</AuthProvider>
    </QueryClientProvider>
  );
}
