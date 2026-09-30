import { useEffect, useState } from "react";

type Health = {
  status: string;
  version: string;
  offline: boolean;
  data_source: "none" | "synthetic" | "uploaded";
};

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("theme", dark ? "dark" : "light");
    } catch {
      /* private mode: ignore */
    }
  }, [dark]);
  return { dark, toggle: () => setDark((d) => !d) };
}

export default function App() {
  const { dark, toggle } = useTheme();
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setHealth)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Persistent synthetic-data banner (Phase 6 wires it to the loaded dataset). */}
      {health?.data_source === "synthetic" && (
        <div className="bg-amber-400 text-amber-950 text-center font-semibold py-2 px-4">
          Evaluated on synthetic data — results do not reflect real component populations.
        </div>
      )}

      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-800">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Burn-in Screening</h1>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Dynamic outlier detection and drift prediction — SIH 2026 PS 26170
          </p>
        </div>
        <button
          onClick={toggle}
          className="rounded-md border border-slate-300 dark:border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-100 dark:hover:bg-slate-800"
          aria-label="Toggle colour theme"
        >
          {dark ? "Light theme" : "Dark theme"}
        </button>
      </header>

      <main className="flex-1 px-6 py-8 max-w-5xl w-full mx-auto space-y-6">
        <section className="rounded-lg border border-slate-200 dark:border-slate-800 p-5">
          <h2 className="text-lg font-semibold mb-2">Phase 0 — scaffold</h2>
          <p className="text-slate-700 dark:text-slate-300">
            No data loaded. The lot overview, trajectory view, flagged-parts table and part detail
            arrive in Phase 6.
          </p>
        </section>

        <section className="rounded-lg border border-slate-200 dark:border-slate-800 p-5">
          <h2 className="text-lg font-semibold mb-2">API connection</h2>
          {health && (
            <dl className="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 font-mono text-sm">
              <dt className="text-slate-500">status</dt>
              <dd className="text-emerald-600 dark:text-emerald-400">{health.status}</dd>
              <dt className="text-slate-500">version</dt>
              <dd>{health.version}</dd>
              <dt className="text-slate-500">offline</dt>
              <dd>{String(health.offline)}</dd>
              <dt className="text-slate-500">data_source</dt>
              <dd>{health.data_source}</dd>
            </dl>
          )}
          {error && (
            <p className="text-rose-600 dark:text-rose-400 text-sm">
              Could not reach the API ({error}). Start it with <code>make api</code>.
            </p>
          )}
          {!health && !error && <p className="text-sm text-slate-500">Connecting…</p>}
        </section>
      </main>

      <footer className="px-6 py-3 text-xs text-slate-500 border-t border-slate-200 dark:border-slate-800">
        Team SYNTROPIC · runs fully offline · no telemetry
      </footer>
    </div>
  );
}
