#!/usr/bin/env bash
# Local video -> original GVHMR weights -> GMR -> G1 kinematic preview.
set -euo pipefail
V2M_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POLICIES_ROOT=$(dirname "$V2M_ROOT")
SUPER_REPO="$POLICIES_ROOT/SuperMimic-Repro-Sim2Real"
export GVHMR_ROOT="$V2M_ROOT/GVHMR"
export GMR_ROOT="$V2M_ROOT/GMR"
export SMPLX_BODY_MODELS="$V2M_ROOT/models"
export G1_XML="$POLICIES_ROOT/g1_deploy/mjlab-assets/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
export GVHMR_DEVICE=${GVHMR_DEVICE:-cpu}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="/home/yiannis/natalia/mamba/envs/vlaclient/bin:$PATH"
VIDEO_INPUT=${1:-$POLICIES_ROOT/IMG_3307.mp4}
VIDEO_INPUT=$(realpath "$VIDEO_INPUT")
CLIP_NAME=$(basename "${VIDEO_INPUT%.*}")
V2M_OUTPUT="$V2M_ROOT/work/main"
mkdir -p "$V2M_OUTPUT/video30" "$V2M_OUTPUT/gvhmr" "$V2M_OUTPUT/preview"
exec > >(tee -a "$V2M_OUTPUT/pipeline.log") 2>&1
"$V2M_ROOT/gvhmr_env/bin/python" "$V2M_ROOT/manage_video_cache.py" "$VIDEO_INPUT" "$V2M_OUTPUT"
VIDEO_30="$V2M_OUTPUT/video30/$CLIP_NAME.mp4"
if [ ! -f "$VIDEO_30" ]; then
  ffmpeg -nostdin -loglevel error -i "$VIDEO_INPUT" -vf fps=30 -an -c:v libx264 -crf 17 -pix_fmt yuv420p "$VIDEO_30"
fi
cd "$GVHMR_ROOT"
"$V2M_ROOT/gvhmr_env/bin/python" tools/demo/demo.py --video "$VIDEO_30" --output_root "$V2M_OUTPUT/gvhmr" -s --no-render
for FLOOR_MODE in drift constant; do
  "$V2M_ROOT/gmr_env/bin/python" "$SUPER_REPO/tools/video2motion/gvhmr_to_supermimic.py" \
    --gvhmr_pred "$V2M_OUTPUT/gvhmr/$CLIP_NAME/hmr4d_results.pt" \
    --name "$CLIP_NAME" --floor_mode "$FLOOR_MODE" --out_dir "$V2M_OUTPUT/csv_$FLOOR_MODE" \
    --robot_xml "$G1_XML" --init_from_human --restart_tol 0.1
done
"$POLICIES_ROOT/g1_deploy/venv/bin/python" - \
  "$V2M_OUTPUT/csv_drift/$CLIP_NAME.csv" "$V2M_OUTPUT/npz/$CLIP_NAME.npz" "$G1_XML" "$SUPER_REPO" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[4]) / "src"))
from supermimic.data.motion_convert import load_g1_model, convert_csv_to_npz
print(convert_csv_to_npz(sys.argv[1], sys.argv[2], 30, 50, model=load_g1_model(sys.argv[3])))
PY
"$V2M_ROOT/gmr_env/bin/python" "$SUPER_REPO/tools/video2motion/render_preview.py" \
  "$V2M_OUTPUT/csv_drift/$CLIP_NAME.csv" "$VIDEO_30" "$V2M_OUTPUT/preview/$CLIP_NAME.mp4" \
  --robot_xml "$G1_XML" --height 640
echo "Preview: $V2M_OUTPUT/preview/$CLIP_NAME.mp4"
ffmpeg -nostdin -loglevel error -i "$V2M_OUTPUT/preview/$CLIP_NAME.mp4" -an \
  -c:v libvpx -b:v 2M -crf 10 -deadline good -cpu-used 4 -pix_fmt yuv420p -y "$V2M_OUTPUT/preview/$CLIP_NAME.webm"
"$V2M_ROOT/gvhmr_env/bin/python" "$V2M_ROOT/manage_video_cache.py" "$VIDEO_INPUT" "$V2M_OUTPUT" --verify
