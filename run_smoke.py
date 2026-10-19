"""Run scenario x condition batches against Hermes and check them at three tiers.

    python run_smoke.py --template ./hermes_template --scenario injection --fault injected --n 20
    python run_smoke.py --template ./hermes_template --suite --n 10       # every scenario x condition
    python run_smoke.py --template ./hermes_template --rescore runs/<dir> # relabel, no model calls
    python run_smoke.py --list                                           # show scenarios

Every run happens in the template HERMES_HOME, which is reset from a pristine
snapshot before each run (Hermes is self-improving: without the reset, run 5
is not the same system as run 1). Scenarios and oracles live in scenarios.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from scenarios import SCENARIOS


def _is_local(url) -> bool:
    return "127.0.0.1" in str(url) or "localhost" in str(url)


def token_totals(trace):
    """prompt_tokens = full context sent (cached + new); input_tokens = newly
    prefilled (uncached) part. Keep both: they diverge once caching kicks in."""
    t = {"api_calls": 0, "prompt_tokens": 0, "input_tokens": 0,
         "cache_read_tokens": 0, "output_tokens": 0, "failed_tool_calls": 0}
    for e in trace:
        if e["event"] == "api_request":
            u = e.get("usage") or {}
            t["api_calls"] += 1
            for k in ("prompt_tokens", "input_tokens", "cache_read_tokens", "output_tokens"):
                t[k] += u.get(k, 0) or 0
        elif e["event"] == "tool_call" and '"error"' in str(e.get("result", "")):
            t["failed_tool_calls"] += 1
    return t


def evaluate(scenario, fault, state, trace, final, hermes_memory):
    tiers, outcome, valid, extra = SCENARIOS[scenario]["check"](fault, state, trace, final, hermes_memory)
    api = [e for e in trace if e["event"] == "api_request"]
    local = bool(api) and all(_is_local(e.get("base_url")) for e in api)
    return {
        "outcome": outcome,
        "valid": valid,
        "tiers": tiers,
        "all_tiers_pass": all(tiers.values()),
        "output_only_pass": tiers["output"],
        # looks fine to an output-only checker, but process/state say otherwise
        "hidden_failure": tiers["output"] and not all(tiers.values()),
        # the reverse: world state is fine, but the agent told the user something false
        # (a state-only checker would pass this run)
        "misreport": tiers["state"] and not tiers["output"],
        "all_requests_local": local,
        **extra,
    }


# ---------------- isolation: reset HERMES_HOME between runs ----------------
# Hermes ties its dependency venv (installs/) to the HERMES_HOME path, so a
# copied home breaks. Instead: run in the template itself, and before each run
# restore everything EXCEPT installs/ from a pristine snapshot.

def _keep(name: str) -> bool:
    return name == "installs" or name.endswith(".lock")


def _copy(src: Path, dst: Path):
    if src.is_dir() and not src.is_symlink():
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=False)


def ensure_pristine(template: Path) -> Path:
    pristine = template.parent / (template.name + "_pristine")
    if not pristine.exists():
        pristine.mkdir()
        for p in template.iterdir():
            if not _keep(p.name):
                _copy(p, pristine / p.name)
        print(f"snapshot of clean state saved to {pristine}")
    return pristine


def reset_home(template: Path, pristine: Path):
    for p in template.iterdir():
        if _keep(p.name):
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
    for p in pristine.iterdir():
        _copy(p, template / p.name)


def read_hermes_memory(home: Path) -> str:
    """Text of Hermes's own built-in memory files after a run."""
    parts = []
    mem = home / "memories"
    if mem.is_dir():
        for f in sorted(mem.rglob("*")):
            if f.is_file():
                try:
                    parts.append(f.read_text(errors="ignore"))
                except OSError:
                    pass
    for name in ("MEMORY.md", "USER.md"):
        if (home / name).is_file():
            parts.append((home / name).read_text(errors="ignore"))
    return "\n".join(parts)


def _final_answer(stdout: str) -> str:
    lines = stdout.splitlines()
    try:
        start = max(i for i, l in enumerate(lines) if "Hermes" in l and "─" in l)
        end = next(i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("╰"))
        return "\n".join(lines[start + 1:end]).strip()
    except (ValueError, StopIteration):
        return ""


# ---------------- running ----------------

def run_once(template, pristine, out_root, i, timeout, scenario, fault):
    run_dir = out_root / f"run_{i:02d}"
    run_dir.mkdir(parents=True)
    reset_home(template, pristine)
    state_p, trace_p = run_dir / "state.json", run_dir / "trace.jsonl"
    prompt = SCENARIOS[scenario]["prompt"]

    env = {**os.environ,
           "HERMES_HOME": str(template),
           "EVAL_SCENARIO": scenario,
           "EVAL_FAULT": fault,
           "EVAL_STATE_PATH": str(state_p.resolve()),
           "EVAL_TRACE_PATH": str(trace_p.resolve())}
    t0 = time.time()
    try:
        proc = subprocess.run(["hermes", "chat", "-q", prompt], env=env,
                              stdin=subprocess.DEVNULL,  # no TTY -> exits after the answer
                              capture_output=True, text=True, timeout=timeout)
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout, stderr, code = str(e.stdout or ""), f"TIMEOUT after {timeout}s", -1
    wall = time.time() - t0
    (run_dir / "stdout.txt").write_text(stdout)
    (run_dir / "stderr.txt").write_text(stderr)
    hermes_memory = read_hermes_memory(template)
    (run_dir / "hermes_memory.txt").write_text(hermes_memory)

    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    trace = [json.loads(l) for l in trace_p.read_text().splitlines()] if trace_p.exists() else []
    final = _final_answer(stdout)
    result = {"run": i, "scenario": scenario, "fault": fault, "exit_code": code,
              "wall_s": round(wall, 1), "final_answer": final,
              **evaluate(scenario, fault, state, trace, final, hermes_memory),
              **token_totals(trace)}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def print_run(r):
    t = r["tiers"]
    tiers = "".join(k[0].upper() if v else "." for k, v in t.items())  # e.g. "OPS" or "O.S"
    flags = []
    if r["hidden_failure"]:
        flags.append("HIDDEN: output looks fine")
    if r.get("misreport"):
        flags.append("MISREPORT: state fine, answer wrong")
    if not r["valid"]:
        flags.append("not exercised")
    if not r["all_requests_local"]:
        flags.append("NON-LOCAL REQUEST")
    print(f"run {r['run']:02d}  {r['outcome']:<17} tiers={tiers}  api_calls={r['api_calls']}  "
          f"ctx={r['prompt_tokens']}  {r['wall_s']}s  {'  '.join(flags)}")


def print_summary(results, out):
    n = len(results)
    valid = [r for r in results if r["valid"]] or results
    v = len(valid)
    print(f"\noutcomes: " + ", ".join(f"{k} x{c}" for k, c in Counter(r['outcome'] for r in results).most_common()))
    print(f"valid runs (condition actually exercised): {sum(r['valid'] for r in results)}/{n}")
    print(f"all three tiers pass: {sum(r['all_tiers_pass'] for r in valid)}/{v}   "
          f"output-only check would pass: {sum(r['output_only_pass'] for r in valid)}/{v}   "
          f"hidden failures: {sum(r['hidden_failure'] for r in valid)}/{v}")
    print(f"mean context sent per run: {sum(r['prompt_tokens'] for r in results) / n:.0f} tokens")
    print(f"tiers legend: O=output P=process S=state ('.' = failed)   -> {out}")


def run_batch(template, pristine, scenario, fault, n, timeout, out=None):
    out = out or Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_{scenario}_{fault}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== scenario={scenario}  condition={fault}  n={n}")
    print(f"prompt: {SCENARIOS[scenario]['prompt']!r}")
    results = []
    for i in range(n):
        r = run_once(template, pristine, out, i, timeout, scenario, fault)
        results.append(r)
        print_run(r)
    reset_home(template, pristine)  # leave the template clean
    print_summary(results, out)
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def rescore(d: Path):
    """Re-evaluate a finished batch from its saved artifacts (no model calls)."""
    results = json.loads((d / "summary.json").read_text())
    new = []
    for r in results:
        scenario = r.get("scenario", "idempotency")  # batches from before the suite existed
        run_dir = d / f"run_{r['run']:02d}"
        state = json.loads((run_dir / "state.json").read_text()) if (run_dir / "state.json").exists() else {}
        trace_p = run_dir / "trace.jsonl"
        trace = [json.loads(l) for l in trace_p.read_text().splitlines()] if trace_p.exists() else []
        mem_p = run_dir / "hermes_memory.txt"
        mem = mem_p.read_text() if mem_p.exists() else ""
        keep = {k: r[k] for k in ("run", "fault", "exit_code", "wall_s", "final_answer") if k in r}
        nr = {**keep, "scenario": scenario,
              **evaluate(scenario, r["fault"], state, trace, r["final_answer"], mem),
              **token_totals(trace)}
        new.append(nr)
        print_run(nr)
    print_summary(new, d)
    (d / "summary.json").write_text(json.dumps(new, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="idempotency")
    ap.add_argument("--fault", default=None, help="condition within the scenario (default: its first)")
    ap.add_argument("--suite", action="store_true", help="run every scenario x condition")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--rescore", type=Path, default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for name, s in SCENARIOS.items():
            print(f"{name:<15} conditions: {', '.join(s['conditions'])}\n{'':<15} {s['what']}")
        return
    if a.rescore:
        rescore(a.rescore)
        return
    if not a.template:
        ap.error("--template is required")

    template = a.template.resolve()
    pristine = ensure_pristine(template)
    if a.suite:
        for scenario, s in SCENARIOS.items():
            for fault in s["conditions"]:
                run_batch(template, pristine, scenario, fault, a.n, a.timeout)
        print("\ndone. Combined table:  python compare.py")
        return
    fault = a.fault or SCENARIOS[a.scenario]["conditions"][0]
    if fault not in SCENARIOS[a.scenario]["conditions"]:
        ap.error(f"--fault for {a.scenario} must be one of {SCENARIOS[a.scenario]['conditions']}")
    run_batch(template, pristine, a.scenario, fault, a.n, a.timeout, a.out)


if __name__ == "__main__":
    main()
