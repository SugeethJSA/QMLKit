import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "QMLKit Kennel Console",
  description:
    "Live kennel telemetry, micro-movement diagnostics and data collection for QMLKit",
};

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/diagnostics", label: "Diagnostics" },
  { href: "/collect", label: "Data Lab" },
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-[#1f2937] px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xl font-bold">QMLKit</span>
            <span className="quantum-badge">Kennel Live</span>
            <span className="text-sm text-gray-400 hidden sm:inline">
              Micro-movement screening · PS-26139
            </span>
          </div>
          <nav className="flex gap-1">
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="px-3 py-2 rounded-lg text-sm text-gray-300 hover:text-white hover:bg-[#1f2937] transition"
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </header>
        <main className="p-6">{children}</main>
      </body>
    </html>
  );
}
