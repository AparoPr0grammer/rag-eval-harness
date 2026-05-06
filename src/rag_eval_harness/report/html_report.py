"""Self-contained HTML report renderer.

Takes one or more run summaries and emits a single HTML file with no external
dependencies — open it in a browser, attach it to a PR, drop it on a static
host. When given multiple summaries, renders a side-by-side comparison table
across every shared metric: this is the BM25-vs-hybrid-vs-dense view.

Accepts dict-shaped data (i.e. the result of ``dataclasses.asdict`` on a
:class:`RunSummary`) so the same renderer works whether you've just finished
a run or you're loading historical summaries from disk.
"""

from __future__ import annotations

import datetime as dt
import html
import math
from collections.abc import Mapping, Sequence
from typing import Any


def render_html_report(
    summaries: Sequence[Mapping[str, Any]],
    *,
    title: str = "RAG Evaluation Report",
) -> str:
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    comparison = _render_comparison(summaries) if len(summaries) > 1 else ""
    pipeline_blocks = "\n".join(_render_pipeline(s) for s in summaries)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_STYLES}</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="meta">Generated {html.escape(generated_at)} · {len(summaries)} pipeline{'' if len(summaries) == 1 else 's'}</div>
{comparison}
{pipeline_blocks}
</body>
</html>"""


_STYLES = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.5; }
h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.25rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.4rem; }
h3 { font-size: 1rem; margin-top: 1.5rem; }
.meta { color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; font-size: 0.9rem; }
th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #eee; }
th { background: #f7f7f9; font-weight: 600; }
.num { font-variant-numeric: tabular-nums; text-align: right; }
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; background: #eef; color: #224; margin-right: 0.4rem; }
.pipeline { border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1.5rem; background: #fafafa; }
details { margin-top: 0.5rem; }
summary { cursor: pointer; color: #555; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; }
.stat { background: white; border: 1px solid #e3e3e3; border-radius: 6px; padding: 0.6rem 0.8rem; }
.stat .label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }
.stat .value { font-size: 1.4rem; font-variant-numeric: tabular-nums; margin-top: 0.2rem; }
@media (prefers-color-scheme: dark) {
  body { background: #1a1a1a; color: #ddd; }
  h2 { border-bottom-color: #333; }
  th { background: #222; }
  .pipeline { background: #1f1f1f; border-color: #333; }
  .stat { background: #222; border-color: #333; }
  .stat .label { color: #999; }
  .badge { background: #234; color: #aef; }
  th, td { border-bottom-color: #333; }
}
"""


def _render_comparison(summaries: Sequence[Mapping[str, Any]]) -> str:
    metrics: dict[str, dict[str, float]] = {}
    for s in summaries:
        pipeline_name = s.get("pipeline", {}).get("name", "?")
        for r in s.get("reports", []):
            for a in r.get("aggregate", []):
                metric_name = a.get("metric", "")
                metrics.setdefault(metric_name, {})[pipeline_name] = a.get("value", float("nan"))

    pipeline_names = [s.get("pipeline", {}).get("name", "?") for s in summaries]
    metric_keys = sorted(metrics.keys())

    rows = []
    for m in metric_keys:
        cells = "".join(
            f'<td class="num">{_format_number(metrics[m].get(p))}</td>' for p in pipeline_names
        )
        rows.append(f"<tr><td>{html.escape(m)}</td>{cells}</tr>")

    headers = "".join(f'<th class="num">{html.escape(p)}</th>' for p in pipeline_names)
    return f"""
<h2>Pipeline comparison</h2>
<table>
  <thead><tr><th>Metric</th>{headers}</tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def _render_pipeline(summary: Mapping[str, Any]) -> str:
    pipeline = summary.get("pipeline", {})
    name = pipeline.get("name", "?")
    description = pipeline.get("description")
    reports = summary.get("reports", [])
    started_at = summary.get("started_at", "?")
    completed_at = summary.get("completed_at", "?")
    results = summary.get("results", [])

    headline_stats: list[str] = []
    for r in reports:
        for a in list(r.get("aggregate", []))[:4]:
            headline_stats.append(_render_stat(a))

    report_sections = "\n".join(_render_report_section(r) for r in reports)

    desc_html = f"<p>{html.escape(description)}</p>" if description else ""
    return f"""
<div class="pipeline">
  <h2>{html.escape(str(name))}</h2>
  {desc_html}
  <div class="meta">Started {html.escape(str(started_at))} · Completed {html.escape(str(completed_at))} · {len(results)} item{'' if len(results) == 1 else 's'}</div>
  <div class="grid">{''.join(headline_stats)}</div>
  {report_sections}
</div>"""


def _render_stat(score: Mapping[str, Any]) -> str:
    metric = html.escape(str(score.get("metric", "")))
    value = _format_number(score.get("value"))
    return (
        f'<div class="stat"><div class="label">{metric}</div>'
        f'<div class="value">{value}</div></div>'
    )


def _render_report_section(report: Mapping[str, Any]) -> str:
    aggregate_rows = "".join(
        f'<tr><td>{html.escape(str(a.get("metric", "")))}</td>'
        f'<td class="num">{_format_number(a.get("value"))}</td></tr>'
        for a in report.get("aggregate", [])
    )
    per_item = list(report.get("per_item", []))
    item_rows = "".join(
        f'<tr><td>{html.escape(str(i.get("item_id", "")))}</td>'
        f'<td class="num">{_format_number(i.get("value"))}</td></tr>'
        for i in per_item[:25]
    )
    name = html.escape(str(report.get("evaluator", "")))
    return f"""
<h3><span class="badge">{name}</span></h3>
<table>
  <thead><tr><th>Metric</th><th class="num">Value</th></tr></thead>
  <tbody>{aggregate_rows}</tbody>
</table>
<details>
  <summary>Per-item scores (first 25 of {len(per_item)})</summary>
  <table>
    <thead><tr><th>Item ID</th><th class="num">Score</th></tr></thead>
    <tbody>{item_rows}</tbody>
  </table>
</details>"""


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    abs_v = abs(v)
    if abs_v >= 1000:
        return f"{v:.0f}"
    if abs_v >= 1:
        return f"{v:.2f}"
    if v == 0:
        return "0"
    return f"{v:.4f}"
