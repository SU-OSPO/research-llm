#!/bin/bash
# deploy_fixes.sh
#
# Installs the fixed files into ~/rag_cluster, backs up the originals, runs
# a quick sanity check, and prints the exact submit commands to run next.
#
# Run from wherever you unpacked cluster_bundle_fixed.tar.gz — this script
# figures out its own directory. It does NOT submit any jobs. That stays
# manual so you decide when.

set -e

CLUSTER_DIR="${HOME}/rag_cluster"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)

echo "==> Target:    ${CLUSTER_DIR}"
echo "==> Source:    ${SCRIPT_DIR}"
echo "==> Timestamp: ${STAMP}"
echo

# --- sanity: files present in SCRIPT_DIR ------------------------------------
for f in run_one_permutation.sh benchmark_parallel.sub benchmark_retry.sub aggregate_parallel.py; do
    if [[ ! -f "${SCRIPT_DIR}/${f}" ]]; then
        echo "ERROR: missing ${SCRIPT_DIR}/${f}" >&2
        exit 1
    fi
done

mkdir -p "${CLUSTER_DIR}"

# --- backup + install -------------------------------------------------------
backup_dir="${CLUSTER_DIR}/_backup_${STAMP}"
mkdir -p "${backup_dir}"

for f in run_one_permutation.sh benchmark_parallel.sub aggregate_parallel.py; do
    if [[ -f "${CLUSTER_DIR}/${f}" ]]; then
        cp -a "${CLUSTER_DIR}/${f}" "${backup_dir}/${f}"
        echo "  backed up: ${f}  ->  _backup_${STAMP}/${f}"
    fi
    cp -a "${SCRIPT_DIR}/${f}" "${CLUSTER_DIR}/${f}"
    echo "  installed: ${f}"
done

# benchmark_retry.sub is new, no original to back up
cp -a "${SCRIPT_DIR}/benchmark_retry.sub" "${CLUSTER_DIR}/benchmark_retry.sub"
echo "  installed: benchmark_retry.sub"

chmod +x "${CLUSTER_DIR}/run_one_permutation.sh"

# --- verify -----------------------------------------------------------------
echo
echo "==> Installed submit-file resource requests:"
grep -E "^request|^\+" "${CLUSTER_DIR}/benchmark_parallel.sub" | sed 's/^/    /'
echo
echo "==> Installed wrapper env exports:"
grep -E "^export " "${CLUSTER_DIR}/run_one_permutation.sh" | sed 's/^/    /'

# --- env check --------------------------------------------------------------
echo
echo "==> Checking conda env 'rag'"
if ! command -v conda >/dev/null 2>&1; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate rag
python -c "import torch; print('    torch:', torch.__version__, 'built for CUDA:', torch.version.cuda)"
python -c "
import importlib, sys
for mod in ('transformers', 'bitsandbytes', 'accelerate', 'chromadb', 'sentence_transformers'):
    try:
        m = importlib.import_module(mod)
        print(f'    {mod}: {getattr(m, \"__version__\", \"?\")}')
    except Exception as e:
        print(f'    {mod}: NOT INSTALLED  ({e})')
" 2>&1

echo
echo "========================================================================="
echo "  DONE. Next steps:"
echo "========================================================================="
echo
echo "  (Option A) Re-run only the 16 previously-failed permutations:"
echo
echo "    cd ~/rag_cluster"
echo "    RUN_TAG=\$(date +%Y%m%d_%H%M%S)_retry"
echo "    echo \"\$RUN_TAG\" > .last_run_tag"
echo "    condor_submit benchmark_retry.sub -append \"run_tag = \$RUN_TAG\""
echo
echo "  (Option B) Re-run ALL 27 for a clean single-run dataset:"
echo
echo "    cd ~/rag_cluster"
echo "    RUN_TAG=\$(date +%Y%m%d_%H%M%S)"
echo "    echo \"\$RUN_TAG\" > .last_run_tag"
echo "    condor_submit benchmark_parallel.sub -append \"run_tag = \$RUN_TAG\""
echo
echo "  Then watch (Ctrl+C exits the watch, jobs keep running):"
echo
echo "    watch -n 10 'condor_q arapte -af:h ClusterId ProcId JobStatus RemoteHost'"
echo
echo "  When done, merge (supports multi-dir for combining original + retry):"
echo
echo "    python aggregate_parallel.py bench_results/parallel_\$RUN_TAG"
echo "    cat bench_results/parallel_\$RUN_TAG/benchmark_summary_merged.txt"
