"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  ArrowRight,
  Cpu,
  Database,
  FlaskConical,
  Eye,
} from "lucide-react";

const NAV = [
  { href: "/", label: "Dashboard", icon: Activity },
  { href: "/diagnostics", label: "Diagnostics", icon: Eye },
  { href: "/train", label: "Training Studio", icon: Cpu },
  { href: "/collect", label: "Data Lab", icon: Database },
  { href: "/lab", label: "Hybrid Lab", icon: FlaskConical },
];

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen flex-col">
      <header className="app-header">
        <div className="mx-auto flex w-full max-w-[1280px] flex-wrap items-center justify-between gap-3 px-5 py-3 sm:px-8">
          <div className="flex items-center gap-5">
            <Link href="/" className="flex items-center gap-2.5" aria-label="QMLKit home">
              <span className="brand-mark" aria-hidden="true">Q</span>
              <span className="brand">QMLKit</span>
              <span className="chip">PS-26139</span>
            </Link>
            <span className="muted hidden text-[13px] lg:inline">
              Hybrid quantum screening console
            </span>
          </div>

          <div className="flex items-center gap-3">
            <nav aria-label="Main navigation" className="flex flex-wrap items-center gap-1">
              {NAV.map(({ href, label, icon: Icon }) => {
                const active = pathname === href;
                return (
                  <Link
                    key={href}
                    href={href}
                    aria-current={active ? "page" : undefined}
                    className={`nav-link ${active ? "nav-active" : ""}`}
                  >
                    <Icon size={16} strokeWidth={1.8} aria-hidden="true" />
                    <span className="hidden sm:inline">{label}</span>
                  </Link>
                );
              })}
            </nav>
            <Link href="/collect" className="btn-primary !hidden sm:!inline-flex">
              New session
              <ArrowRight size={15} strokeWidth={2} aria-hidden="true" />
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1280px] flex-1 px-5 py-8 sm:px-8 sm:py-10">
        {children}
      </main>

      <footer className="app-footer">
        <span>QMLKit · PS-26139 · canine biomimetic VOC &amp; micro-movement analysis</span>
        <span>Research aid — not a diagnosis</span>
      </footer>
    </div>
  );
}
