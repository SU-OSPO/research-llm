#!/usr/bin/env python3
"""
aggregate_parallel.py — Merge per-permutation benchmark JSONs into one file.

Fixes vs the previous version:
  - Dedupes on (model_key, db_key). If multiple runs exist for the same
    permutation, prefers a successful one over a failed one; otherwise the
    most recently modified file wins.
  - Flags the 27 - N missing permutations explicitly so you can see at a
    glance what still needs a re-run.
  - Can merge from MULTIPLE source directories in one shot (e.g. the original
    run + a retry run): pass them all as args and the dedup logic picks the
    successful record across runs.

Usage:
    python aggregate_parallel.py bench_results/parallel_<run_tag>
    python aggregate_parallel.py bench_results/parallel_A bench_results/parallel_B

Output (written to the LAST directory argument):
    <dir>/benchmark_results_merged.json
    <dir>/benchmark_summary_merged.txt
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# The 27 expected permutations — used to report what's missing.
EXPECTED_MODELS = [
    "llama-3.2-3b", "gemma-4-e2b", "gemma-4-e4b", "llama-3.1-8b",
    "qwen-2.5-14b", "gpt-oss-20b", "gemma-4-26b", "gemma-4-31b", "llama-3.3-70b",
]
EXPECTED_DBS = ["full", "openalex", "abstracts"]


def _is_success(run: Dict[str, Any]) -> bool:
    err = (run.get("error") or "").strip()
    qs = run.get("questions", [])
    return bool(qs) and not err


def _collect_runs(directories: List[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return a dict keyed by (model_key, db_key) holding the best run for that pair.
    Better = successful > failed; if both same status, newer file wins.
    """
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    best_mtime: Dict[Tuple[str, str], float] = {}

    for directory in directories:
        if not directory.is_dir():
            print(f"  [WARN] skipping missing dir {directory}")
            continue
        print(f"Scanning {directory}/ ...")
        for path in sorted(directory.glob("benchmark_results_*.json")):
            if "merged" in path.name:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [WARN] failed to read {path.name}: {e}")
                continue
            mtime = path.stat().st_mtime
            for run in data.get("runs", []):
                key = (run.get("model_key", "?"), run.get("db_key", "?"))
                existing = best.get(key)
                take = False
                if existing is None:
                    take = True
                else:
                    was_success = _is_success(existing)
                    now_success = _is_success(run)
                    if now_success and not was_success:
                        take = True
                    elif now_success == was_success and mtime > best_mtime[key]:
                        take = True
                if take:
                    best[key] = run
                    best_mtime[key] = mtime
            n_runs = len(data.get("runs", []))
            print(f"  [OK]   {path.name}  ({n_runs} run{'s' if n_runs != 1 else ''})")

    return best


def _merge(directories: List[Path]) -> Dict[str, Any]:
    best = _collect_runs(directories)

    # Deterministic ordering: model order as listed, then db order as listed.
    def sort_key(r: Dict[str, Any]) -> Tuple[int, int]:
        m = r.get("model_key", "")
        db = r.get("db_key", "")
        mi = EXPECTED_MODELS.index(m) if m in EXPECTED_MODELS else len(EXPECTED_MODELS)
        di = EXPECTED_DBS.index(db) if db in EXPECTED_DBS else len(EXPECTED_DBS)
        return (mi, di)

    runs = sorted(best.values(), key=sort_key)

    # Aggregate stats
    total_time_s = 0.0
    gpus_seen = set()
    success = fail = 0
    for r in runs:
        for q in r.get("questions", []):
            total_time_s += float(q.get("timing", {}).get("total_ms", 0) or 0) / 1000.0
        for g in (r.get("gpu") or []):
            gpus_seen.add(str(g))
        if _is_success(r):
            success += 1
        else:
            fail += 1

    # Missing permutations (no run at all)
    missing = [
        {"model_key": m, "db_key": db}
        for m in EXPECTED_MODELS for db in EXPECTED_DBS
        if (m, db) not in best
    ]

    tag_suffix = directories[-1].name
    merged = {
        "benchmark_tag": f"merged_{tag_suffix}",
        "source_dirs": [str(d) for d in directories],
        "status": "complete" if (success == len(EXPECTED_MODELS) * len(EXPECTED_DBS)) else "partial",
        "progress": f"{success}/{len(EXPECTED_MODELS) * len(EXPECTED_DBS)}",
        "successful_permutations": success,
        "failed_permutations": fail,
        "missing_permutations": missing,
        "total_time_s": round(total_time_s, 2),
        "note": "merged from parallel HTCondor jobs — total_time_s is sum of "
                "per-question wall times, not wall-clock duration",
        "gpus_observed": sorted(gpus_seen),
        "config": {
            "models": sorted({r.get("model_key", "") for r in runs if r.get("model_key")}),
            "databases": sorted({r.get("db_key", "") for r in runs if r.get("db_key")}),
            "question_count": max((len(r.get("questions", [])) for r in runs), default=0),
        },
        "runs": runs,
    }
    return merged


def _write_summary(merged: Dict[str, Any], out_path: Path) -> None:
    runs = merged["runs"]
    lines = []
    lines.append(f"Merged Benchmark Summary — {merged['benchmark_tag']}")
    lines.append(f"Source dirs: {', '.join(merged.get('source_dirs', []))}")
    lines.append(f"Progress: {merged['progress']}  (success={merged['successful_permutations']}, "
                 f"fail={merged['failed_permutations']}, missing={len(merged.get('missing_permutations', []))})")
    lines.append(f"Total question wall-time: {merged['total_time_s']:.0f}s")
    if merged.get("gpus_observed"):
        lines.append(f"GPUs observed: {', '.join(merged['gpus_observed'])}")
    lines.append("")
    lines.append(f"{'Model':<22} {'Database':<12} {'Q#':>3} {'Errors':>7} {'AvgGen(ms)':>12} {'Status':<8}")
    lines.append("-" * 80)

    for r in runs:
        model = (r.get("model_label") or r.get("model_key", "?"))[:20]
        db = (r.get("db_key") or "?")[:12]
        qs = r.get("questions", [])
        errs = sum(1 for q in qs if q.get("error"))
        if r.get("error"):
            errs += 1
        gens = [float(q.get("timing", {}).get("generation_ms", 0) or 0)
                for q in qs if q.get("answer_word_count", 0) > 5]
        avg_gen = sum(gens) / len(gens) if gens else 0
        status = "OK" if _is_success(r) else "FAIL"
        lines.append(f"{model:<22} {db:<12} {len(qs):>3} {errs:>7} {avg_gen:>12.0f} {status:<8}")
        if r.get("error"):
            lines.append(f"  ERROR: {r['error'][:100]}")

    if merged.get("missing_permutations"):
        lines.append("")
        lines.append("MISSING (no JSON file, not even a failure record):")
        for m in merged["missing_permutations"]:
            lines.append(f"  {m['model_key']:<18s} x {m['db_key']}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    directories = [Path(a) for a in sys.argv[1:]]
    for d in directories:
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}")
            return 1

    merged = _merge(directories)

    out_dir = directories[-1]
    out_json = out_dir / "benchmark_results_merged.json"
    out_txt = out_dir / "benchmark_summary_merged.txt"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    _write_summary(merged, out_txt)

    print()
    print(f"Merged: {merged['successful_permutations']} successful, "
          f"{merged['failed_permutations']} failed, "
          f"{len(merged.get('missing_permutations', []))} missing")
    print(f"  -> {out_json}")
    print(f"  -> {out_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
