#!/usr/bin/env bash
# GPU extraction environment; the released models and GVHMR algorithm stay the same.
set -euo pipefail
TASK_ROOT=${V2M_CLOUD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
command -v uv >/dev/null || { echo 'Install uv in this VM first.' >&2; exit 1; }
command -v c++ >/dev/null || { echo 'A C++ compiler is required for PyTorch3D.' >&2; exit 1; }
uv venv --python 3.10 "$TASK_ROOT/env"
PY="$TASK_ROOT/env/bin/python"
uv pip install --python "$PY" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" 'numpy==1.26.4' 'setuptools<81' pip wheel ninja \
  'timm==0.9.12' 'lightning==2.3.0' 'pytorch-lightning==2.3.0' 'torchmetrics<1.5' \
  'hydra-core==1.3.2' hydra-zen hydra-colorlog rich tqdm 'opencv-python==4.11.0.86' \
  ffmpeg-python scikit-image termcolor einops 'imageio==2.34.1' 'av==13.0.0' \
  joblib trimesh smplx==0.1.28 'ultralytics==8.2.42' lapx tensorboardX fvcore iopath wis3d requests
# Pose extraction uses PyTorch3D's tensor transforms, which execute on CUDA
# through PyTorch. Its optional rasterizer isn't used (--no-render), so a CPU
# extension avoids a separate Blackwell CUDA-extension toolchain requirement.
if [ ! -d "$TASK_ROOT/pytorch3d" ]; then
  git clone --depth 1 --branch v0.7.6 https://github.com/facebookresearch/pytorch3d.git "$TASK_ROOT/pytorch3d"
fi
PYTORCH3D_FORCE_NO_CUDA=1 MAX_JOBS=${MAX_JOBS:-8} \
  uv pip install --python "$PY" --no-build-isolation "$TASK_ROOT/pytorch3d"
uv pip install --python "$PY" --no-deps -e "$TASK_ROOT/GVHMR"
mkdir -p "$TASK_ROOT/GVHMR/inputs/checkpoints"
if [ ! -e "$TASK_ROOT/GVHMR/inputs/checkpoints/body_models" ]; then
  ln -s "$TASK_ROOT/models" "$TASK_ROOT/GVHMR/inputs/checkpoints/body_models"
fi
"$PY" - <<'PY'
import os, subprocess, sys
probe = '''
import torch
from pytorch3d.transforms import matrix_to_axis_angle
assert torch.cuda.is_available(), 'CUDA PyTorch is not working'
x = torch.ones((128,128), device='cuda:0')
assert torch.isfinite(x@x).all()
assert torch.isfinite(matrix_to_axis_angle(torch.eye(3,device='cuda:0'))).all()
print(torch.cuda.get_device_name(0), 'PyTorch', torch.__version__, 'CUDA', torch.version.cuda, flush=True)
'''
for index in os.environ.get('V2M_GPU_IDS', '1,7').split(','):
    uuid = subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    print('Checking physical GPU', index, uuid, flush=True)
    subprocess.run([sys.executable, '-c', probe],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid), check=True)
PY
