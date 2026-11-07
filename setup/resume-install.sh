#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
if ! grep -q '^# R39 .*REVISION: 2\.' /etc/nv_tegra_release; then
  echo 'Requires JetPack 7.2.x / L4T R39.2.x. Upgrade the Jetson first.' >&2
  exit 1
fi
if [[ "$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')" != 3.12 ]]; then
  echo 'Requires system Python 3.12 from the matching JetPack installation.' >&2
  exit 1
fi
cd "$ROOT/Isaac-GR00T"
git lfs install --local
# Download only the small inference demo, excluding unrelated LFS media.
git lfs pull --include='demo_data/libero_demo/**' --exclude=''
bash scripts/deployment/jetson/install_deps.sh
source .venv/bin/activate
source scripts/activate_jetpack72.sh
python -c 'import torch, gr00t; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
echo 'Runtime installed. Authenticate Hugging Face and select a checkpoint before inference.'
