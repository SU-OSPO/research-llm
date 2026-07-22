# Syracuse Research Assistant — HPC Cluster Bundle

A parallel benchmarking and deployment toolkit for running the RAG pipeline at scale on the OrangeGrid HPC cluster. This bundle extends the core RAG application with SLURM job submission scripts, multi-node parallel execution, GPU-specific benchmark configurations, and a unified orchestration layer for permutation sweeps.

This README is intentionally code aware. It explains how each module in the bundle contributes to the cluster workflow, what each script controls, how jobs are submitted and monitored, and how results are aggregated after parallel runs complete.

---

## Table of contents

1. Project summary
2. Main capabilities
3. Repository structure
4. End to end flow
5. File by file technical reference
6. Cluster workflow in plain language
7. SLURM job configuration reference
8. Setup and cluster execution
9. Runtime operations
10. Strengths and limitations
11. Recommended next improvements

---

## Project summary

At a practical level, this bundle solves four problems:

1. It parameterizes RAG benchmark runs as permutation sweeps across model, retrieval, and prompt configuration dimensions, submitting each permutation as an independent SLURM job.
2. It provides GPU hardware-specific job submission templates tuned for A100 and L40S nodes on the OrangeGrid cluster.
3. It aggregates per-permutation benchmark results from distributed output files into a single consolidated report.
4. It handles the operational complexity of cluster execution including job queuing, dependency chaining, sleep-and-wait polling, and retry logic for failed or preempted jobs.

---

## Main capabilities

### Permutation-based benchmarking

The benchmark system enumerates configuration dimensions and submits one SLURM job per permutation. Each job executes an isolated RAG pipeline evaluation so results are independent and the sweep can be parallelized across hundreds of nodes simultaneously.

### Hardware-specific job templates

Separate SLURM submission templates target A100 and L40S GPU nodes. Each template specifies the appropriate GPU resource requests, memory limits, time allocations, and module loads for the hardware profile.

### Sleep-and-wait coordination

Long-running sweeps use a polling mechanism to wait for dependent jobs to complete before downstream steps such as aggregation proceed. The sleep-and-wait layer handles queue delays gracefully without holding a login-node process open.

### Result aggregation

After all permutation jobs complete, a dedicated aggregation script collects per-job output files and produces a unified benchmark report suitable for analysis.

### Retry and resilience

A retry submission script re-queues failed permutations without resubmitting the entire sweep, reducing wasted compute time on transient cluster failures.

---

## Repository structure

```
cluster_bundle/
|-- aggregate_parallel.py
|-- benchmark_a100.sub
|-- benchmark_l40s.sub
|-- benchmark_parallel.sub
|-- benchmark_rag.py
|-- benchmark_retry.sub
|-- config_full.py
|-- conversation_memory.py
|-- database_manager.py
|-- deploy_fixes.sh
|-- orangegrid_commands.txt
|-- PARALLEL_USAGE.md
|-- rag_chat.py
|-- rag_engine.py
|-- rag_pipeline.py
|-- rag_utils.py
|-- README.md
|-- requirements.txt
|-- runtime_settings.py
|-- run_benchmark.sh
|-- run_one_permutation.sh
|-- run_sleep.sh
|-- session_store.py
|-- setup_cluster.sh
|-- sleep_and_wait.py
|-- sleep_and_wait.sub
|-- streamlit_app.py
```

---

## End to end flow

```
flowchart TD
    U[Researcher configures benchmark sweep] --> S1[setup_cluster.sh installs dependencies and verifies environment]
    S1 --> S2[benchmark_parallel.sub submitted to SLURM scheduler]
    S2 --> P1[SLURM allocates GPU node and runs benchmark_rag.py]

    P1 --> PM1[benchmark_rag.py loads config_full.py and runtime_settings.py]
    PM1 --> PM2[Instantiate DatabaseManager and get active Chroma collection]
    PM2 --> PM3[Run permutation sweep over configured parameter dimensions]

    PM3 --> PJ1{Single-node or parallel mode}
    PJ1 -->|parallel| PJ2[run_one_permutation.sh called per permutation]
    PJ2 --> PJ3[benchmark_a100.sub or benchmark_l40s.sub submitted per job]
    PJ3 --> PJ4[Each SLURM job runs isolated rag_pipeline.answer_question call]
    PJ1 -->|single-node| PJ5[benchmark_rag.py runs permutations sequentially]

    PJ4 --> W1[sleep_and_wait.sub polls for job completion]
    PJ5 --> W1

    W1 --> W2[sleep_and_wait.py checks SLURM job states]
    W2 --> W3{All jobs complete}
    W3 -->|no| W4[run_sleep.sh waits and polls again]
    W4 --> W2
    W3 -->|yes| AG1[aggregate_parallel.py collects per-job output files]

    AG1 --> AG2[Parse and normalize result records]
    AG2 --> AG3[Merge into unified benchmark report]
    AG3 --> AG4[Write consolidated output for analysis]

    AG4 --> RT1{Any jobs failed or timed out}
    RT1 -->|yes| RT2[benchmark_retry.sub resubmits failed permutations]
    RT2 --> PJ3
    RT1 -->|no| DONE[Sweep complete]
```

### Job lifecycle

```
flowchart LR
    Q[benchmark_parallel.sub submitted] --> S[SLURM queues job and allocates GPU resources]
    S --> R[run_one_permutation.sh called with permutation index]
    R --> E[benchmark_rag.py evaluates one permutation]
    E --> O[Output written to per-job result file]
    O --> A[aggregate_parallel.py reads all result files]
    A --> N[Next sweep configured if needed]
```

---

## File by file technical reference

## 1. `benchmark_rag.py`

This script is the primary benchmark driver. It evaluates the RAG pipeline across one or more configuration permutations and records output metrics.

### Main responsibilities

1. Load runtime configuration from `config_full.py` and `runtime_settings.py`.
2. Construct the list of permutations from configured parameter axes.
3. For each permutation, initialize the pipeline and execute a set of benchmark queries.
4. Capture retrieval metrics, answer latency, confidence labels, and output quality indicators.
5. Write per-permutation results to a structured output file.

### Important implementation notes

`benchmark_rag.py` is designed to be invoked both interactively and from within a SLURM job step via `run_one_permutation.sh`. When called from a SLURM step it expects the permutation index to be passed as a command line argument so the script can select the correct parameter combination from the precomputed permutation list.

---

## 2. `aggregate_parallel.py`

This script collects distributed per-job output files after a parallel benchmark sweep completes and merges them into a single consolidated report.

### Main responsibilities

1. Scan the configured output directory for result files matching the benchmark job naming pattern.
2. Parse each file and normalize field names and value types.
3. Deduplicate results if any permutations were retried and produced multiple output records.
4. Compute aggregate statistics across permutations such as mean latency and retrieval confidence distribution.
5. Write the merged report to a single output file.

### Important implementation notes

`aggregate_parallel.py` is designed to run after all SLURM jobs in a sweep have exited. It should be called from the final step of `sleep_and_wait.py` or invoked manually after verifying job completion through `orangegrid_commands.txt`.

---

## 3. `benchmark_a100.sub`

This is the SLURM submission template for benchmark jobs targeting A100 GPU nodes.

### Key SLURM directives

1. `#SBATCH --gres=gpu:a100` requests an A100 GPU.
2. Memory and CPU allocations are sized for the A100 memory profile.
3. Time limit is set to accommodate full permutation execution including model load time.
4. Output and error log paths are parameterized with the permutation index for per-job diagnostics.

### Usage

Submit directly via `sbatch benchmark_a100.sub` for a single job or call `run_one_permutation.sh` to parameterize and submit programmatically.

---

## 4. `benchmark_l40s.sub`

This is the SLURM submission template for benchmark jobs targeting L40S GPU nodes.

### Key SLURM directives

1. `#SBATCH --gres=gpu:l40s` requests an L40S GPU.
2. Resource allocations reflect L40S VRAM capacity and typical memory bandwidth characteristics.
3. Module loads are adjusted for the software environment available on L40S nodes.

### When to use

Use `benchmark_l40s.sub` when A100 nodes are unavailable or when comparing throughput characteristics between GPU families. The two templates can be submitted in parallel to run the same permutation sweep on both hardware profiles simultaneously.

---

## 5. `benchmark_parallel.sub`

This is the SLURM submission template for the parallel sweep coordinator. It manages the fan-out submission of individual permutation jobs.

### Key SLURM directives

1. Requests a CPU-only login adjacent allocation sufficient to drive job submission without occupying a GPU node.
2. Sets environment variables consumed by `run_one_permutation.sh` to control sweep scope.
3. Chains into `sleep_and_wait.sub` as a dependent step after fan-out completes.

---

## 6. `benchmark_retry.sub`

This SLURM submission template re-queues failed permutations from a prior sweep without resubmitting successful ones.

### Main responsibilities

1. Read the list of failed permutation indices from the sweep output directory.
2. Submit individual job steps for each failed index using the same templates as the original sweep.
3. Re-enter the `sleep_and_wait` polling loop to coordinate aggregation after retries complete.

### Important implementation notes

Retry behavior depends on output files being present for completed permutations. `aggregate_parallel.py` uses presence and completeness of output files to determine which permutations succeeded and which need retry.

---

## 7. `sleep_and_wait.sub`

This SLURM submission template wraps `sleep_and_wait.py` as a job step that polls for sweep completion without occupying a GPU allocation.

### Key behavior

1. Submits as a dependency of the fan-out coordinator.
2. Polls SLURM job state at a configured interval.
3. Exits when all tracked job IDs have completed.
4. On completion, triggers `aggregate_parallel.py` or signals the next pipeline stage.

---

## 8. `sleep_and_wait.py`

This script implements the polling logic used by the sleep-and-wait coordination layer.

### Main responsibilities

1. Accept a list of SLURM job IDs to monitor.
2. Query SLURM for current job states using `sacct` or `squeue`.
3. Sleep for a configured interval between polls to avoid overloading the scheduler.
4. Return success when all monitored jobs have reached a terminal state.
5. Return failure or raise a timeout exception when the maximum wait duration is exceeded.

### Functions

#### `query_job_states(job_ids)`

Calls the SLURM accounting tools to retrieve current state for each tracked job ID and returns a mapping of job ID to state string.

#### `all_terminal(states)`

Returns true when all state values in the mapping correspond to terminal SLURM states such as COMPLETED, FAILED, CANCELLED, or TIMEOUT.

#### `wait_for_jobs(job_ids, poll_interval_s, max_wait_s)`

Main polling loop. Sleeps between polls and exits when `all_terminal` returns true or the maximum wait is exceeded.

---

## 9. `run_one_permutation.sh`

This shell script wraps a single permutation execution. It is called by the parallel coordinator or `benchmark_parallel.sub` with the permutation index as an argument.

### Main responsibilities

1. Set the permutation index environment variable consumed by `benchmark_rag.py`.
2. Activate the Python virtual environment for the cluster execution context.
3. Execute `benchmark_rag.py` with appropriate flags for single-permutation mode.
4. Write a completion marker file to the output directory so `aggregate_parallel.py` can verify coverage.

---

## 10. `run_benchmark.sh`

This shell script provides a convenience entry point for running the full benchmark sweep from a single command on the cluster login node.

### Main responsibilities

1. Validate that the environment is configured and dependencies are present.
2. Submit `benchmark_parallel.sub` to the SLURM scheduler.
3. Print the assigned job ID and a link to the relevant log path.

---

## 11. `run_sleep.sh`

This shell script wraps a single iteration of the sleep-and-wait polling cycle for use inside SLURM job steps where Python is not the primary executor.

---

## 12. `setup_cluster.sh`

This script prepares the cluster execution environment for a fresh deployment or after a Python environment update.

### Main responsibilities

1. Create or recreate the virtual environment in the cluster scratch or home directory.
2. Install dependencies from `requirements.txt`.
3. Set environment variables for local model paths, Chroma directories, and SQLite database location.
4. Run a quick smoke test to verify the RAG pipeline imports and Chroma client initializes.
5. Print a summary of configured paths and available GPU resources.

---

## 13. `deploy_fixes.sh`

This script applies targeted fixes to the cluster deployment without requiring a full environment rebuild.

### Main responsibilities

1. Copy updated Python modules from the bundle to the active deployment directory.
2. Optionally clear Chroma and session caches when breaking changes are deployed.
3. Restart any persistent background processes such as a running Streamlit server.

---

## 14. `orangegrid_commands.txt`

This file is a reference card for commonly used OrangeGrid cluster operations.

### Typical contents

1. Commands for checking queue status and job accounting.
2. Node and partition availability queries.
3. Interactive GPU session request patterns.
4. Log inspection commands for running and completed jobs.
5. Storage quota checks and scratch space management.

This file is a living reference and should be updated as cluster policies and available partitions change.

---

## 15. `PARALLEL_USAGE.md`

This document explains the parallel sweep workflow in end-user terms. It complements this README by providing step-by-step instructions for a researcher who wants to run a new benchmark sweep without detailed knowledge of the underlying orchestration.

### Typical contents

1. How to configure the permutation axes in `benchmark_rag.py`.
2. How to choose between A100 and L40S templates.
3. How to monitor sweep progress using OrangeGrid commands.
4. How to retrieve and interpret aggregated results.
5. How to use the retry submission script when some jobs fail.

---

## 16. `config_full.py`

This file provides environment-driven defaults for the primary retrieval mode. It is shared with the core RAG application and drives both single-node and cluster execution.

### Important values

1. `SQLITE_DB_FULL` — path to the SQLite source database
2. `CHROMA_DIR_FULL` — Chroma persistence directory
3. `CHROMA_COLLECTION_FULL` — active collection name
4. `LLAMA_1B` and `LLAMA_3B` — local model paths used by benchmark runs
5. `EMBED_MODEL` — embedding model path or HuggingFace name
6. `CHUNK_MAX_CHARS`, `PAPERS_PER_BATCH`, and `CHROMA_MAX_BATCH` — ingestion throughput settings

On the cluster these values are typically overridden by environment variables set in `setup_cluster.sh` or the SLURM submission scripts to point at scratch-space paths.

---

## 17. `runtime_settings.py`

This file is the runtime tuning surface of the RAG pipeline. On the cluster it is used by benchmark jobs to configure retrieval budget, prompt sizing, model selection, and memory behavior per permutation.

### Key fields used in cluster benchmarking

1. `active_mode` — which corpus configuration to use
2. `llm_model` — which local model to evaluate
3. `search_k` and `search_fetch_k` — retrieval budget dimensions swept across permutations
4. `prompt_max_docs` and `prompt_doc_text_limit` — prompt packing dimensions swept across permutations
5. `answer_max_new_tokens` — generation budget
6. `llm_timeout_s` — timeout guard particularly important in cluster environments with variable GPU load

---

## 18. `database_manager.py`

This file abstracts corpus mode selection. In the cluster context it ensures that all parallel benchmark jobs connect to the same Chroma collection rather than creating competing client instances.

### Cluster-relevant behavior

`ensure_dirs_exist()` is called by `setup_cluster.sh` to create Chroma directories on scratch space before any benchmark jobs are submitted.

---

## 19. `conversation_memory.py`

This file handles in-process answer caching and pipeline caching. In the cluster benchmark context the caching behavior is typically disabled or scoped to a single permutation so results across jobs remain independent and comparable.

---

## 20. `session_store.py`

This file is the durable session state store backed by SQLite. In stateless benchmark mode each permutation job creates an isolated session to avoid cross-contamination of conversation state between permutations.

---

## 21. `rag_engine.py`, `rag_pipeline.py`, `rag_utils.py`

These three files are the core RAG pipeline modules shared with the main application. See the main `rag/README.md` for complete function-level documentation. In the cluster bundle they are included to make the benchmark jobs self-contained without requiring the jobs to import from the parent application directory.

---

## 22. `rag_chat.py`

This file provides a lightweight command-line chat interface to the RAG pipeline. In the cluster context it is used for quick interactive testing after `setup_cluster.sh` completes, before a full benchmark sweep is submitted.

---

## 23. `streamlit_app.py`

This file is the Streamlit UI from the main application. It is included in the cluster bundle so that a live serving instance can be started on a GPU node for interactive use alongside the benchmark infrastructure.

---

## 24. `requirements.txt`

This file lists the Python dependencies required for the cluster execution environment.

### Installing on the cluster

```bash
module load python/3.10
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU-specific torch builds may require an additional index URL. Consult the current OrangeGrid documentation for the recommended PyTorch wheel source for the available CUDA version.

---

## Cluster workflow in plain language

### Submitting a benchmark sweep

1. Configure permutation axes in `benchmark_rag.py`, choosing the model keys, retrieval budgets, and prompt sizes to sweep.
2. Set environment variables for database and model paths in `setup_cluster.sh` or export them directly.
3. Run `setup_cluster.sh` to verify the environment.
4. Run `run_benchmark.sh` to submit the parallel coordinator to SLURM.
5. Monitor progress with the commands in `orangegrid_commands.txt`.

### Aggregating results

After all jobs complete, run `aggregate_parallel.py` directly or wait for it to be triggered automatically by `sleep_and_wait.py`:

```bash
python aggregate_parallel.py --output-dir /path/to/benchmark/outputs
```

### Handling failures

If some jobs failed or timed out:

```bash
sbatch benchmark_retry.sub
```

The retry script reads which permutations are missing from the output directory and resubmits only those.

---

## SLURM job configuration reference

| Template | Target GPU | Typical time limit | Use case |
|---|---|---|---|
| `benchmark_a100.sub` | A100 | 4–8 hours | Primary benchmark runs |
| `benchmark_l40s.sub` | L40S | 4–8 hours | Hardware comparison or A100 unavailability |
| `benchmark_parallel.sub` | CPU | 30 minutes | Sweep coordination and fan-out |
| `benchmark_retry.sub` | Inherits | Variable | Resubmitting failed permutations |
| `sleep_and_wait.sub` | CPU | 12 hours | Post-fan-out polling and aggregation trigger |

---

## Setup and cluster execution

### Prerequisites

1. OrangeGrid cluster account with GPU partition access
2. Python 3.10 or newer available via module system
3. SQLite database with populated `research_info` and `works` tables accessible from cluster scratch space
4. Pre-built Chroma index or access to run `chroma_ingest.py` from the parent application
5. Local model weights for the configured `LLAMA_1B` and `LLAMA_3B` paths available on shared storage

### Initial setup

```bash
# Clone or copy the bundle to your cluster home or scratch directory
cd /path/to/cluster_bundle

# Configure paths
export SQLITE_DB_FULL=/scratch/<netid>/research.db
export CHROMA_DIR_FULL=/scratch/<netid>/chroma_full
export LLAMA_3B=/scratch/<netid>/models/llama-3b

# Run setup
bash setup_cluster.sh
```

### Submitting the benchmark

```bash
bash run_benchmark.sh
```

### Monitoring

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

Refer to `orangegrid_commands.txt` for the full reference of available monitoring commands.

### Retrieving results

After the sweep completes, results are written to the output directory configured in `benchmark_rag.py`. Run `aggregate_parallel.py` to merge them:

```bash
python aggregate_parallel.py
```

---

## Runtime operations

### Why permutation jobs are independent

Each benchmark job writes results to a unique output file and uses an isolated session store. This means any subset of jobs can be rerun or retried without corrupting other results. `aggregate_parallel.py` uses output file presence to determine sweep coverage.

### Why sleep-and-wait exists instead of SLURM dependencies

SLURM job dependency chains are brittle when individual jobs in a large array fail. The sleep-and-wait approach polls job state externally and proceeds to aggregation only when all tracked jobs have exited, regardless of whether some exited with failure codes. Failed jobs are identified separately by `benchmark_retry.sub`.

### Why two GPU templates exist

A100 and L40S nodes have different memory capacities and software environments on OrangeGrid. Maintaining separate templates avoids hard-coding GPU-specific flags in the main benchmark script and makes it easy to run the same permutation sweep on both hardware profiles.

### Why `deploy_fixes.sh` exists separately from `setup_cluster.sh`

`setup_cluster.sh` is destructive and takes several minutes to recreate the virtual environment. `deploy_fixes.sh` applies targeted file replacements for bug fixes during active benchmarking without requiring environment recreation or resubmitting in-progress jobs.

### Why model paths are environment variables rather than hardcoded

Cluster storage paths for large model weights vary by user and allocation. Externalizing paths to environment variables allows the same submission scripts to work across different user accounts and storage layouts without modification.





