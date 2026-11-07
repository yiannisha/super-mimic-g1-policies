#!/usr/bin/env bash
set -euo pipefail
source /home/yiannis/policies/g1_deploy/env.sh
if (( $# )); then
  python /home/yiannis/policies/g1_deploy/motion_suite/prepare_bones.py "$@"
else
  python /home/yiannis/policies/g1_deploy/motion_suite/prepare_bones_parallel.py
fi
/home/yiannis/policies/g1_deploy/run-motion.sh list
