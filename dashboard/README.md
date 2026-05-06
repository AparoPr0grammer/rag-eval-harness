# rag-eval dashboard

A small Next.js app for browsing the JSON `RunSummary` files emitted by the harness.

## Quickstart

```bash
cd dashboard
npm install
mkdir -p runs
cp ../run.json runs/2026-05-06-baseline.json
npm run dev
```

Open http://localhost:3000.

## How it works

The dashboard reads JSON files from a directory (default `./runs/`). Drop any number of `RunSummary` files in there — they appear on the homepage, sorted newest-first by filename. Naming runs like `YYYY-MM-DD-<label>.json` keeps them in a sensible order.

Configure the directory via:

```bash
RAG_EVAL_RUNS_DIR=/path/to/runs npm run dev
```

## What it shows

- **Homepage** — every run, with the headline metric from each evaluator (faithfulness, context precision, etc.) shown as chips.
- **Run detail** — full aggregate scores per evaluator, plus the first 25 items with their generated answers.

## Architecture

The dashboard is intentionally minimal:

- **No database.** JSON files on disk, read at request time.
- **No auth.** Run it locally or behind your own reverse proxy.
- **No upload UI.** Either drop files into the directory, or rsync them from CI.
- **No client-side state.** Server components, a tiny bit of CSS.

The point is to be a *reader*, not a workflow tool. The Python harness does the work; the dashboard is the window onto its output.

## Production use

For a team setup, point `RAG_EVAL_RUNS_DIR` at a shared storage location (S3 with a fuse mount, an NFS share, a synced Dropbox folder), have CI write `RunSummary` JSON there after every evaluation run, and `next build && next start` the dashboard wherever you host internal tools.
