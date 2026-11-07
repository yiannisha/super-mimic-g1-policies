#!/usr/bin/env bash
set -euo pipefail
source /home/yiannis/policies/g1_deploy/env.sh
exec python /home/yiannis/policies/SuperMimic-Repro-Sim2Real/scripts/deploy/bridge/policy_control.py \
  --socket /home/yiannis/policies/g1_deploy/control.sock "$@"
