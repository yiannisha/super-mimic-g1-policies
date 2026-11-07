#!/usr/bin/env bash
set -euo pipefail
source /home/yiannis/policies/g1_deploy/env.sh
exec python /home/yiannis/policies/g1_deploy/motion_suite/launcher.py "$@"
