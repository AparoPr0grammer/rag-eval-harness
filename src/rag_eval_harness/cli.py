"""Command-line entry point.

Two commands:

    rag-eval synthesise <markdown-dir> <output.jsonl>
        Generate a Q/A test set from a directory of markdown files.

    rag-eval report <run.json>... -o <out.html>
        Render an HTML report from one or more saved run summaries.

The CLI deliberately doesn't try to drive a full evaluation — that needs a
user-supplied retriever and generator, which only makes sense in code. See
the ``examples/`` directory for end-to-end runners.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .report.html_report import render_html_report
from .synthetic.generator import SyntheticTestSetGenerator


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Tools for the rag-eval-harness Python package.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "synthesise",
        aliases=["synthesize"],
        help="Generate a Q/A test set from a directory of markdown files",
    )
    s.add_argument("directory", help="Directory of .md / .mdx files to walk")
    s.add_argument("output", help="Path to output JSONL file")

    r = sub.add_parser(
        "report",
        help="Render an HTML report from one or more run summary JSON files",
    )
    r.add_argument("inputs", nargs="+", help="Path(s) to run.json files")
    r.add_argument("-o", "--output", required=True, help="Path to output HTML file")
    r.add_argument("-t", "--title", default="RAG Evaluation Report", help="Report title")

    args = parser.parse_args()
    if args.command in ("synthesise", "synthesize"):
        asyncio.run(_synthesise(args.directory, args.output))
    elif args.command == "report":
        _report(args.inputs, args.output, args.title)


async def _synthesise(directory: str, output: str) -> None:
    gen = SyntheticTestSetGenerator()
    print(f"Reading markdown from {directory}...")
    items = await gen.from_directory(directory)
    SyntheticTestSetGenerator.write_jsonl(items, output)
    print(f"Wrote {len(items)} synthetic items to {output}")


def _report(inputs: list[str], output: str, title: str) -> None:
    summaries = []
    for inp in inputs:
        with Path(inp).open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    html = render_html_report(summaries, title=title)
    Path(output).write_text(html, encoding="utf-8")
    print(f"Wrote HTML report to {output}")


if __name__ == "__main__":
    main()
