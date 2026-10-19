"""Combine every batch in runs/ into one results table, one row per scenario x condition.

    python compare.py            # all labelled batches
    python compare.py --min-n 20 # only batches with at least 20 runs

Columns
  valid            runs where the condition was actually exercised
  all_tiers        output + process + state all correct (the real pass rate)
  output_only      what an answer-only checker would report
  hidden           output looked fine but process/state failed (output-only checker misses it)
  state_only       what a state-only checker would report
  misreport        state fine but the answer was false (state-only checker misses it)
  outcomes         label counts
Writes results.csv as well.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--min-n", type=int, default=1)
    a = ap.parse_args()

    groups = {}
    for s in sorted(a.runs.glob("*/summary.json")):
        rs = json.loads(s.read_text())
        if len(rs) < a.min_n or not rs or "tiers" not in rs[0]:
            continue  # skip batches not yet (re)scored with three tiers
        key = (rs[0].get("scenario", "idempotency"), rs[0].get("fault", "none"))
        groups.setdefault(key, []).extend(rs)

    if not groups:
        print("no three-tier batches found (run --rescore on older batches first)")
        return

    rows = []
    for (scen, cond), rs in sorted(groups.items()):
        valid = [r for r in rs if r["valid"]]
        v = len(valid) or 1
        pct = lambda k: f"{sum(r[k] for r in valid) / v:.0%}"
        row = {
            "scenario": scen, "condition": cond, "runs": len(rs), "valid": len(valid),
            "all_tiers": pct("all_tiers_pass"),
            "output_only": pct("output_only_pass"),
            "hidden": pct("hidden_failure"),
            "state_only": f"{sum(r['tiers']['state'] for r in valid) / v:.0%}",
            "misreport": f"{sum(r['tiers']['state'] and not r['tiers']['output'] for r in valid) / v:.0%}",
            "mean_ctx_tokens": round(sum(r["prompt_tokens"] for r in valid) / v),
            "mean_api_calls": round(sum(r["api_calls"] for r in valid) / v, 2),
            "outcomes": "; ".join(f"{k}:{c}" for k, c in Counter(r["outcome"] for r in rs).most_common()),
        }
        if scen == "lost_update":
            row["outcomes"] += f" | ran_concurrently:{sum(r.get('ran_concurrently', False) for r in rs)}"
        if scen == "idempotency":
            row["outcomes"] += f" | harness_retries:{sum('harness' in r.get('retry_source', '') for r in rs)}"
        rows.append(row)

    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("\nrates are over valid runs; saved to results.csv")


if __name__ == "__main__":
    main()
