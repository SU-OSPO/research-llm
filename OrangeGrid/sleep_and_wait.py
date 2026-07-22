#!/usr/bin/env python3
"""Sleep job — prints GPU/env info, then sleeps so you can condor_ssh_to_job in."""
import os, time, subprocess, sys

print("=" * 60)
print("HOSTNAME:", os.environ.get("HOSTNAME", "unknown"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "none"))
print("Python:", sys.executable, sys.version)
print("=" * 60)

# GPU info
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception:
    print("nvidia-smi failed")

# PyTorch / CUDA
try:
    import torch
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
except Exception as e:
    print(f"torch check failed: {e}")

# Model files
print("\n--- Models ---")
models_dir = os.path.expanduser("~/models")
if os.path.isdir(models_dir):
    for d in sorted(os.listdir(models_dir)):
        p = os.path.join(models_dir, d)
        if os.path.isdir(p):
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, fns in os.walk(p) for f in fns) / 1e9
            print(f"  {d}: {size:.1f} GB")
else:
    print(f"  {models_dir} not found!")

# Data files
print("\n--- Data ---")
data_dir = os.path.expanduser("~/research-llm-data")
if os.path.isdir(data_dir):
    for d in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, d)
        if os.path.isdir(p):
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, fns in os.walk(p) for f in fns) / 1e9
            print(f"  {d}: {size:.1f} GB")
else:
    print(f"  {data_dir} not found!")

print("\n" + "=" * 60)
print("Sleeping 4 hours — use condor_ssh_to_job to connect...")
print("=" * 60)
time.sleep(14400)
