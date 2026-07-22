#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag
cd ~/rag_cluster

echo "=========================================="
echo "  RAG BENCHMARK — Native-First Precision"
echo "  $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU')"
echo "=========================================="

python benchmark_rag.py --output-dir ~/rag_cluster/bench_results 2>&1 | tee benchmark_live.log

echo "=========================================="
echo "  BENCHMARK COMPLETE — $(date)"
echo "=========================================="
