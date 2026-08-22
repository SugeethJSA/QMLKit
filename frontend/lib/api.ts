/** API base URL auto-detection with ?api= override (repomono convention). */

export function apiBase(): string {
  if (typeof window !== "undefined") {
    const override = new URLSearchParams(window.location.search).get("api");
    if (override) return override.replace(/\/$/, "");
    return `http://${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
}

export function wsUrl(path: string): string {
  return apiBase().replace(/^http/, "ws") + path;
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`);
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return res.json();
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail ?? `POST ${path} -> ${res.status}`);
  }
  return res.json();
}
