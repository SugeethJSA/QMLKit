import type { Metadata } from "next";
import Link from "next/link";
import { Activity, BarChart3, Cpu, Database, Eye, FlaskConical, Sparkles } from "lucide-react";
import "./globals.css";

export const metadata: Metadata = {
  title: "QMLKit - Quantum Machine Learning Cancer Detection Platform",
  description: "Hybrid Quantum-Classical Disease Detection & Canine Olfactory Screening Portal (Problem Statement 26139)",
};

const NAV = [
  { href: "/", label: "Dashboard", icon: Activity },
  { href: "/diagnostics", label: "Diagnostics", icon: Eye },
  { href: "/train", label: "Training Studio", icon: Cpu, badge: "QML" },
  { href: "/collect", label: "Data Lab", icon: Database },
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased bg-[#07090e] text-gray-100 flex flex-col">
        {/* Top High-Tech Navbar */}
        <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#0b0f19]/85 border-b border-white/10 px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-500/20 group-hover:scale-105 transition-transform">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-bold tracking-tight text-white group-hover:text-purple-300 transition-colors">
                    QMLKit
                  </span>
                  <span className="quantum-badge text-[10px]">PS-26139</span>
                </div>
              </div>
            </Link>
            <span className="text-xs text-gray-400 hidden md:inline-block border-l border-white/10 pl-3">
              Hybrid Quantum Machine Learning Platform for Early Disease Detection
            </span>
          </div>

          <div className="flex items-center gap-3">
            <nav className="flex items-center gap-1 bg-[#07090e]/60 p-1 rounded-xl border border-white/5">
              {NAV.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="px-3.5 py-1.5 rounded-lg text-xs font-semibold text-gray-300 hover:text-white hover:bg-white/10 transition flex items-center gap-1.5 relative"
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{item.label}</span>
                    {item.badge && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded-full bg-purple-500/30 text-purple-300 border border-purple-500/40">
                        {item.badge}
                      </span>
                    )}
                  </Link>
                );
              })}
            </nav>

            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/5 border border-white/5 text-xs text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[11px] font-medium text-gray-300">Simulator Active</span>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">{children}</main>

        {/* Footer */}
        <footer className="border-t border-white/5 px-6 py-4 text-center text-xs text-gray-400 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>QMLKit · Problem Statement 26139 · Canine Biomimetic VOC &amp; Micro-Movement Analysis</span>
          <span className="font-mono text-[11px] text-gray-400">PennyLane · Qiskit · PyTorch · Scikit-Learn</span>
        </footer>
      </body>
    </html>
  );
}
