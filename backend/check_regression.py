"""
check_regression.py — Phase 3 (Regression Gating) for CI.

Pulls recent traced RAG queries from LangSmith and fails (non-zero exit)
if citation coverage or P95 latency has regressed past a set threshold.
Designed to run as a step in GitHub Actions, right alongside the
Project 1 RAGAS eval gate.

Usage:
    python check_regression.py
"""

import os
import sys
import statistics
from dotenv import load_dotenv

load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "my-local-rag")

# ── Thresholds — tune these as you gather more real traffic ─────
MIN_CITATION_COVERAGE_PCT = 70.0
MAX_P95_LATENCY_SECONDS   = 15.0
MIN_RUNS_REQUIRED         = 5   # don't judge on too little data

if not LANGCHAIN_API_KEY:
    print("[ERROR] LANGCHAIN_API_KEY is not set. Add it to your .env / CI secrets.")
    sys.exit(1)

from langsmith import Client

client = Client(api_key=LANGCHAIN_API_KEY)

runs = list(client.list_runs(
    project_name=LANGCHAIN_PROJECT,
    filter='eq(name, "rag_query")',
    is_root=True,
    limit=50,
))
print("\n--- Per-run breakdown ---")
for r in runs:
    if r.start_time and r.end_time:
        dur = (r.end_time - r.start_time).total_seconds()
        print(f"Run {r.id} | {dur:.2f}s | started at {r.start_time}")
        
if not runs:
    print("[WARN] No traces found. Skipping regression check (nothing to compare).")
    sys.exit(0)

if len(runs) < MIN_RUNS_REQUIRED:
    print(f"[WARN] Only {len(runs)} runs found (need {MIN_RUNS_REQUIRED}+ for a reliable check).")
    print("[WARN] Skipping regression gate — not enough traffic yet.")
    sys.exit(0)

# ── Citation coverage ────────────────────────────────────────────
answered_count = 0
for r in runs:
    meta = (r.extra or {}).get("metadata", {})
    if meta.get("answered"):
        answered_count += 1

total = len(runs)
coverage_pct = (answered_count / total * 100) if total else 0

# ── P95 latency ───────────────────────────────────────────────────
latencies = []
for r in runs:
    if r.start_time and r.end_time:
        latencies.append((r.end_time - r.start_time).total_seconds())

p95 = None
if latencies:
    latencies_sorted = sorted(latencies)
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]

print(f"[INFO] Citation coverage : {coverage_pct:.1f}% ({answered_count}/{total})")
print(f"[INFO] P95 latency       : {p95:.2f}s" if p95 is not None else "[INFO] P95 latency       : no data")
# ── Regression gate ────────────────────────────────────────────
failures = []

if coverage_pct < MIN_CITATION_COVERAGE_PCT:
    failures.append(
        f"Citation coverage {coverage_pct:.1f}% is below minimum {MIN_CITATION_COVERAGE_PCT}%"
    )

if p95 is not None and p95 > MAX_P95_LATENCY_SECONDS:
    failures.append(
        f"P95 latency {p95:.2f}s exceeds maximum {MAX_P95_LATENCY_SECONDS}s"
    )

print()
if failures:
    print("❌ REGRESSION CHECK FAILED:")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)   # non-zero exit = CI step fails
else:
    print("✅ REGRESSION CHECK PASSED — no thresholds breached.")
    sys.exit(0)

print(f"[INFO] Pulled {len(runs)} recent runs from project '{LANGCHAIN_PROJECT}'")