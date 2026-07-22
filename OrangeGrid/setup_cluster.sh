#!/bin/bash
set -e

echo "=== Setting up RAG cluster environment ==="

# Activate conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
else
    echo "ERROR: Miniconda not found at ~/miniconda3"
    echo "Install it first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3"
    echo "  ~/miniconda3/bin/conda init bash && source ~/.bashrc"
    exit 1
fi

# Create or update conda env
if conda env list | grep -q "^rag "; then
    echo "[1/3] Conda env 'rag' exists — updating packages..."
    conda activate rag
else
    echo "[1/3] Creating conda env 'rag' with Python 3.11..."
    conda create -y -n rag python=3.11
    conda activate rag
fi

# Install Python deps
echo "[2/3] Installing Python dependencies..."
pip install -r requirements.txt

# NLTK data
echo "[3/3] Downloading NLTK data..."
python -c "
import nltk
for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng', 'maxent_ne_chunker',
            'maxent_ne_chunker_tab', 'words', 'stopwords', 'names',
            'wordnet', 'omw-1.4']:
    try: nltk.download(pkg, quiet=True)
    except: pass
print('NLTK data OK')
"

# Create output dirs
mkdir -p ~/rag_cluster/bench_results

# Make scripts executable
chmod +x ~/rag_cluster/run_benchmark.sh
chmod +x ~/rag_cluster/run_sleep.sh

echo ""
echo "=== Setup complete ==="
echo "Models dir:  ~/models/"
echo "Data dir:    ~/research-llm-data/"
echo "Code dir:    ~/rag_cluster/"
echo "Results dir: ~/rag_cluster/bench_results/"
