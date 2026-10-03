import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";

import Providers from "./providers";
import "./globals.css";

// next/font downloads and self-hosts these at build time -- no runtime request
// to a font CDN, so the app renders identically offline and there is no
// third-party origin in the critical path.
const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-plex-sans",
  display: "swap",
});

// Every figure in this product is a measurement someone may read against
// another: RMS against RMS, risk against risk. They get a mono face with
// tabular figures so columns line up and a ticking counter does not reflow.
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "FleetMind — Predictive Maintenance",
  description:
    "Federated Intelligent Predictive Maintenance for Bearing Fault Diagnosis",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable}`}>
      <body className="min-h-screen bg-canvas font-sans text-ink antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
