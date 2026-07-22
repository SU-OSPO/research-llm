#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag
cd ~/rag_cluster
python sleep_and_wait.py
