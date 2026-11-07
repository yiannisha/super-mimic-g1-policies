#!/usr/bin/env bash
# Run one clip per visible GPU. Outputs are copied back for GMR and deployment simulation.
set -euo pipefail
TASK_ROOT=${V2M_CLOUD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CLIP=${1:?clip name required}
GPU_INDEX=${2:?GPU index or UUID required}
# NVML indices and CUDA ordinals can differ when another card is unavailable.
# Resolve the requested physical card before importing PyTorch.
GPU_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export GVHMR_DEVICE=cuda
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
# These are our existing, checksum-verified legacy checkpoints. Preserve their
# original torch.load behavior when using modern PyTorch on Blackwell.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$TASK_ROOT/GVHMR"
exec "$TASK_ROOT/env/bin/python" tools/demo/demo.py \
  --video "$TASK_ROOT/inputs/$CLIP.mp4" --output_root "$TASK_ROOT/work/gvhmr" -s --no-render
