import Link from "next/link";
import { listRuns } from "@/lib/runs";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const runs = await listRuns();

  if (runs.length === 0) {
    return (
      <div className="empty">
        <p>No runs found.</p>
        <p>
          Drop one or more <code>RunSummary</code> JSON files into{" "}
          <code>./runs/</code> to see them here, or set{" "}
          <code>RAG_EVAL_RUNS_DIR</code> to point elsewhere.
        </p>
        <pre>cp ../run.json runs/2026-05-06-bm25-baseline.json</pre>
      </div>
    );
  }

  return (
    <>
      <p className="muted">
        {runs.length} run{runs.length === 1 ? "" : "s"}
      </p>
      {runs.map((run) => (
        <div key={run.id} className="card">
          <h2 style={{ margin: "0 0 0.4rem", fontSize: "1.1rem" }}>
            <Link href={`/runs/${run.id}`}>{run.pipelineName}</Link>
          </h2>
          <div className="row">
            <div className="muted">{formatDate(run.startedAt)}</div>
            <div className="muted">
              {run.itemCount} item{run.itemCount === 1 ? "" : "s"}
            </div>
            <div className="muted">{run.filename}</div>
          </div>
          <div style={{ marginTop: "0.6rem" }}>
            {run.headlineMetrics.map((m) => (
              <span key={m.metric} className="metric">
                {m.metric}: {formatValue(m.value)}
              </span>
            ))}
          </div>
        </div>
      ))}
    </>
  );
}

function formatValue(v: number): string {
  if (Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(4);
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
