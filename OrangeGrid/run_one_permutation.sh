#!/bin/bash
# run_one_permutation.sh — node-local everything edition
#
# Runs ONE (model × db) permutation. Called by benchmark_parallel.sub for
# each queued job.
#
# Args:
#   $1 = model_key  (e.g. llama-3.2-3b)
#   $2 = db_key     (full | openalex | abstracts)
#   $3 = run_tag    (unique tag shared by all jobs in this run)
#
# The cluster's ~/research-llm-data lives on NFS (10.5.0.205:/arapte).
# ChromaDB over NFS is catastrophically slow: a single HNSW query issues
# thousands of random reads, each with NFS round-trip latency. One test
# job spent 43 minutes on Q1 just paging the openalex index over NFS.
#
# Fix: at job start, rsync the specific ChromaDB directory this job needs
# to /tmp (node-local disk). All subsequent queries hit local storage.
# Staging cost: ~30-120s depending on DB size. Query speedup: ~100-1000x.
#
# Also staged to /tmp:
#   - RAG_STATE_DB (conversation sqlite, per-job unique)
#   - RAG_MEMORY_DIR (chroma memory store, per-job unique)
#   - RAG_CACHE_DIR (generic cache, per-job unique)
#   - HF caches (HF_HOME, TRANSFORMERS_CACHE, HF_DATASETS_CACHE)
# Plus RAG_LLM_TIMEOUT_S=600 to lift the outer subprocess cap to ~5.3 h.

set -e
set -o pipefail

MODEL_KEY="$1"
DB_KEY="$2"
RUN_TAG="${3:-parallel_run}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag
cd ~/rag_cluster

# --- unique per-job scratch on node-local /tmp ------------------------------

JOB_ID="${_CONDOR_SLOT_NAME:-${HOSTNAME}}_$$_$(date +%s%N)"
JOB_ID="${JOB_ID//[^A-Za-z0-9_-]/_}"
SCRATCH="/tmp/rag_${USER}_${JOB_ID}"
mkdir -p "${SCRATCH}"
trap 'rm -rf "${SCRATCH}"' EXIT

# --- stage the ChromaDB for this job's DB key -------------------------------

NFS_DATA="${HOME}/research-llm-data"
case "${DB_KEY}" in
    full)      NFS_CHROMA="${NFS_DATA}/chroma_store_full"   ;;
    openalex)  NFS_CHROMA="${NFS_DATA}/chroma_db"           ;;
    abstracts) NFS_CHROMA="${NFS_DATA}/chroma_abstracts"    ;;
    *) echo "ERROR: unknown db_key '${DB_KEY}'" >&2; exit 1 ;;
esac

if [[ ! -d "${NFS_CHROMA}" ]]; then
    echo "ERROR: chroma dir missing on NFS: ${NFS_CHROMA}" >&2
    exit 1
fi

LOCAL_CHROMA="${SCRATCH}/chroma"
mkdir -p "${LOCAL_CHROMA}"

# Point config_full.py at the local copy for THE db this job actually uses.
# The other two stay pointed at NFS (they're never opened in this process).
case "${DB_KEY}" in
    full)      export CHROMA_DIR_FULL="${LOCAL_CHROMA}"      ;;
    openalex)  export CHROMA_DIR_OPENALEX="${LOCAL_CHROMA}"  ;;
    abstracts) export CHROMA_DIR_ABSTRACTS="${LOCAL_CHROMA}" ;;
esac

echo "==> Staging ChromaDB ${DB_KEY} from NFS to local /tmp..."
echo "    src: ${NFS_CHROMA}"
echo "    dst: ${LOCAL_CHROMA}"
echo "    nfs size: $(du -sh "${NFS_CHROMA}" 2>/dev/null | awk '{print $1}')"
STAGE_START=$(date +%s)
rsync -a --no-perms --no-owner --no-group \
      --info=progress2 --no-inc-recursive \
      "${NFS_CHROMA}/" "${LOCAL_CHROMA}/"
STAGE_END=$(date +%s)
echo "==> Staged in $((STAGE_END - STAGE_START))s. Local size: $(du -sh "${LOCAL_CHROMA}" 2>/dev/null | awk '{print $1}')"

# --- per-job state + caches (also /tmp) -------------------------------------

export RAG_STATE_DB="${SCRATCH}/chat_state.sqlite"
export RAG_MEMORY_DIR="${SCRATCH}/chroma_memory"
export RAG_CACHE_DIR="${SCRATCH}/cache"
mkdir -p "${RAG_MEMORY_DIR}" "${RAG_CACHE_DIR}"

export HF_HOME="${SCRATCH}/hf"
export TRANSFORMERS_CACHE="${SCRATCH}/hf/transformers"
export HF_DATASETS_CACHE="${SCRATCH}/hf/datasets"
export HF_HUB_OFFLINE=1
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# Per-question LLM timeout: 600 * 31 + 600 = 19200s outer cap (~5.3 h).
export RAG_LLM_TIMEOUT_S="${RAG_LLM_TIMEOUT_S:-600}"
export TOKENIZERS_PARALLELISM=false

# --- output dir on NFS (small JSON writes are fine over NFS) ----------------

OUTPUT_DIR="bench_results/parallel_${RUN_TAG}"
mkdir -p "${OUTPUT_DIR}"

# --- log header -------------------------------------------------------------

echo "======================================================================"
echo "  PARALLEL BENCHMARK - ${MODEL_KEY} x ${DB_KEY}"
echo "  run_tag:     ${RUN_TAG}"
echo "  host:        $(hostname)"
echo "  started:     $(date)"
echo "  scratch:     ${SCRATCH}"
echo "  chroma:      ${LOCAL_CHROMA}  (staged from ${NFS_CHROMA})"
echo "  state_db:    ${RAG_STATE_DB}"
echo "  memory_dir:  ${RAG_MEMORY_DIR}"
echo "  llm_timeout: ${RAG_LLM_TIMEOUT_S}s per question"
echo "  GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 'no GPU')"
python -c "
import torch
print(f'  torch:       {torch.__version__}  cuda_available={torch.cuda.is_available()}', flush=True)
if torch.cuda.is_available():
    print(f'  device:      {torch.cuda.get_device_name(0)}', flush=True)
"
echo "======================================================================"

# --- run the benchmark ------------------------------------------------------

python benchmark_rag.py \
    --models "${MODEL_KEY}" \
    --databases "${DB_KEY}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/live_${MODEL_KEY}_${DB_KEY}.log"

BENCH_RC=${PIPESTATUS[0]}

echo "======================================================================"
echo "  DONE - ${MODEL_KEY} x ${DB_KEY} - $(date) - rc=${BENCH_RC}"
echo "======================================================================"

exit ${BENCH_RC}
