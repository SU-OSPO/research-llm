# Parallel Benchmark — Quick Reference

Runs all 27 (9 models × 3 databases) permutations simultaneously across
whatever GPU slots HTCondor can match.

## Submit

```bash
cd ~/rag_cluster

# Pick a tag for this run (used to name output dirs + log files)
RUN_TAG=$(date +%Y%m%d_%H%M%S)
echo "Run tag: $RUN_TAG"

# Condor doesn't interpolate env vars inside sub files; pass via a macro
condor_submit benchmark_parallel.sub -append "run_tag = ${RUN_TAG}"
```

This queues 27 jobs. Each requests one GPU. HTCondor matches each to the
smallest free card that satisfies its VRAM requirement:
- 3B / E2B / E4B / 8B → any 22GB+ GPU (RTX 6000, L40S, A40, A100)
- 14B / 20B → 45GB+ (L40S, A40, A100)
- 26B / 31B / 70B → 81GB (A100 80GB only)

Small-model jobs start almost immediately (RTX 6000 pool is huge).
Large-model jobs queue for A100 80GB slots — there are 15 total, currently
6 free.

## Monitor

```bash
# All your jobs
condor_q arapte

# Which hosts each is running on
condor_q arapte -af:h ClusterId ProcId JobStatus RemoteHost

# Live tail of one specific permutation (once it's running)
tail -f ~/rag_cluster/bench_parallel.${RUN_TAG}.gemma-4-31b.full.*.out

# Watch overall progress
watch -n 15 'condor_q arapte | tail -20'
```

Job status codes: `I` = idle (waiting for GPU), `R` = running, `H` = held,
`X` = removed, `C` = completed.

## Merge results (after all jobs finish)

Each job writes its own `benchmark_results_*.json` into
`bench_results/parallel_${RUN_TAG}/`. Merge them into one file:

```bash
python aggregate_parallel.py bench_results/parallel_${RUN_TAG}

# Look at the merged summary
cat bench_results/parallel_${RUN_TAG}/benchmark_summary_merged.txt
```

Copy merged JSON back to your laptop:

```bash
# On Windows, from CMD
scp arapte@its-og-login4.syr.edu:~/rag_cluster/bench_results/parallel_*/benchmark_results_merged.json "C:\Users\arapte\Downloads\"
```

## Kill all parallel jobs (if you need to abort)

```bash
condor_rm arapte
```

## Re-run a single failed permutation

If, say, gemma-4-31b × openalex crashed but everything else finished, just
re-submit that one:

```bash
# Pick any VRAM tier that matches the model (81000 for 31B)
condor_submit -append "model_key = gemma-4-31b" \
              -append "db_key = openalex" \
              -append "run_tag = ${RUN_TAG}_retry" \
              -append "gpu_vram_mb = 81000" \
              -append "queue" \
              benchmark_parallel.sub
```

Or easier — edit the sub file to include only the one permutation and submit.

## How long will this take?

Sequential benchmark on your Windows Turing: ~50 minutes per (model × db),
27 permutations = 22 hours if run serially.

Parallel on cluster with A100 flash-attention:
- 8 small-model jobs (3B, E2B, E4B, 8B on each DB) → start instantly on
  RTX 6000 pool, ~10-15 min each
- 6 mid jobs (14B, 20B on each DB) → start on L40S, ~20-30 min each
- 9 large jobs (26B, 31B, 70B on each DB) → queue for 6 A100 80GB nodes,
  run 6 in parallel + 3 more once some finish; ~30-60 min each

**Total wall time expected: ~1-2 hours** instead of 22, depending on A100
contention.
