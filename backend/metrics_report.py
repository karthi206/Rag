"""
metrics_report.py — Phase 2 (Metrics) deliverable for the Monitoring &
Observability project.

Pulls the most recent traced RAG queries from Langsmith and computes:
  - Citation coverage  (% of queries actually answered vs refused)
  - P50 / P95 latency  (typical vs worst-case response time)
  - Average chunks retrieved / sources cited

Requires LANGCHAIN_API_KEY and LANGCHAIN_PROJECT to be set in .env
(same ones used for tracing).

Usage:
    python metrics_report.py [--limit 50]
"""

import os
import sys
import argparse
import statistics
from dotenv import load_dotenv

load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT  = os.getenv("LANGCHAIN_PROJECT", "my-local-rag")

if not LANGCHAIN_API_KEY:
    print("[ERROR] LANGCHAIN_API_KEY is not set. Add it to your .env file.")
    sys.exit(1)

from langsmith import Client

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=50, help="How many recent runs to analyze")
args = parser.parse_args()

client = Client(api_key=LANGCHAIN_API_KEY)

print("=" * 60)
print("  RAG MONITORING — Metrics Report")
print("=" * 60)
print(f"  Project : {LANGCHAIN_PROJECT}")
print(f"  Analyzing the last {args.limit} 'rag_query' traces...")
print("=" * 60 + "\n")

print("[DEBUG] LANGCHAIN_PROJECT from .env:", repr(LANGCHAIN_PROJECT))
print("[DEBUG] Projects visible to this API key:")
for p in client.list_projects():
    print("   -", repr(p.name))
    
runs = list(client.list_runs(
    project_name=LANGCHAIN_PROJECT,
    filter='eq(name, "rag_query")',
    is_root=True,
    limit=args.limit,
))


if not runs:
    print("[WARN] No traces found yet. Ask your app a few questions first,")
    print("       then run this script again.")
    sys.exit(0)

# ── Latency ──────────────────────────────────────────────────────
latencies = []
for r in runs:
    if r.start_time and r.end_time:
        latencies.append((r.end_time - r.start_time).total_seconds())

# ── Citation coverage + retrieval stats ─────────────────────────
answered_count  = 0
chunks_retrieved = []
sources_cited    = []

for r in runs:
    meta = (r.extra or {}).get("metadata", {})
    if meta.get("answered"):
        answered_count += 1
    if "num_chunks_retrieved" in meta:
        chunks_retrieved.append(meta["num_chunks_retrieved"])
    if "num_sources_cited" in meta:
        sources_cited.append(meta["num_sources_cited"])

total = len(runs)

print(f"📊 Total queries analyzed : {total}\n")

# Citation coverage
coverage_pct = (answered_count / total * 100) if total else 0
print(f"📌 Citation Coverage       : {answered_count}/{total} answered ({coverage_pct:.1f}%)")
print(f"   (refused/'not found'  : {total - answered_count} — this is expected behavior, not a bug,")
print(f"    when the documents genuinely don't cover a question)\n")

# Latency
if latencies:
    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]
    print(f"⏱  Latency")
    print(f"   P50 (typical)         : {p50:.2f}s")
    print(f"   P95 (worst-case)      : {p95:.2f}s")
    print(f"   Min / Max             : {min(latencies):.2f}s / {max(latencies):.2f}s\n")
else:
    print("⏱  Latency: no timing data available on these runs.\n")

# Retrieval stats
if chunks_retrieved:
    print(f"🔎 Avg chunks retrieved per query : {statistics.mean(chunks_retrieved):.1f}")
if sources_cited:
    print(f"📎 Avg sources cited per answer   : {statistics.mean(sources_cited):.1f}")

print("\n" + "=" * 60)
print("  Full trace details: https://smith.langchain.com")
print("=" * 60)
print("[DEBUG] LANGCHAIN_PROJECT from .env:", repr(LANGCHAIN_PROJECT))
print("[DEBUG] Projects visible to this API key:")
for p in client.list_projects():
    print("   -", repr(p.name))