#!/usr/bin/env bash
set -euo pipefail
source /home/yiannis/policies/g1_deploy/env.sh
cd /home/yiannis/policies/SuperMimic-Repro-Sim2Real
export CKPT=e24_0355000
export CLIPS="$PWD/deploy_artifacts/clips/shanshen_duosuomao_lean35kfs_pad2.npz"
export E2E_NAME=local_e24_auto
echo "Diagnostic only: this host previously simulated at 0.14x real time and failed standby." >&2
exec bash scripts/deploy/bridge/run_sim_e2e_bridge.sh auto "$@"
