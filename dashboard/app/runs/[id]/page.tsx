import Link from "next/link";
import { notFound } from "next/navigation";
import { getRun } from "@/lib/runs";

export const dynamic = "force-dynamic";

interface PageProps {
  params: { id: string };
}

export default async function RunDetailPage({ params }: PageProps) {
  const run = await getRun(params.id);
  if (!run) notFound();

  return (
    <>
      <p>
        <Link href="/">← All runs</Link>
      </p>
      <h2 style={{ marginTop: 0 }}>{run.pipeline.name}</h2>
      {run.pipeline.description && <p>{run.pipeline.description}</p>}
      <div className="muted">
        Started {formatDate(run.started_at)} · Completed{" "}
        {formatDate(run.completed_at)} · {run.results.length} items
      </div>

      {run.reports.map((report) => (
        <div key={report.evaluator} className="card">
          <h3 style={{ marginTop: 0 }}>{report.evaluator}</h3>
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th className="num">Value</th>
              </tr>
            </thead>
            <tbody>
              {report.aggregate.map((a) => (
                <tr key={a.metric}>
                  <td>{a.metric}</td>
                  <td className="num">{formatValue(a.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}

      <div className="card">
        <h3 style={{ marginTop: 0 }}>
          Items{" "}
          <span className="muted" style={{ fontSize: "0.85rem", fontWeight: 400 }}>
            (first 25 of {run.results.length})
          </span>
        </h3>
        <table>
          <thead>
            <tr>
              <th style={{ width: "8%" }}>ID</th>
              <th style={{ width: "32%" }}>Question</th>
              <th>Answer</th>
            </tr>
          </thead>
          <tbody>
            {run.results.slice(0, 25).map((r) => (
              <tr key={r.item_id}>
                <td>{r.item_id}</td>
                <td>{r.question}</td>
                <td>{r.generation.text}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function formatValue(v: number): string {
  if (Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
