#!/usr/bin/env bash
# Commissioning INTERP/HOLD only; follow docs/DEPLOY_G1.md before enabling policy.
set -euo pipefail
source /home/yiannis/policies/g1_deploy/env.sh
cd /home/yiannis/policies/SuperMimic-Repro-Sim2Real
exec python scripts/deploy/bridge/g1_runner_bridge.py \
  --iface "${1:-eth0}" --domain 0 --stage air \
  --artifacts deploy_artifacts/e24_0355000 \
  --clip deploy_artifacts/clips/shanshen_duosuomao_lean35kfs_pad2.npz \
  --log-dir deploy_logs/real "${@:2}"
