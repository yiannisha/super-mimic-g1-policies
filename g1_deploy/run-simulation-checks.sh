#!/usr/bin/env bash
set -euo pipefail
source /home/yiannis/policies/g1_deploy/env.sh
cd /home/yiannis/policies/SuperMimic-Repro-Sim2Real
python scripts/deploy/bridge/test_bridge.py | tee "$G1_DEPLOY_HOME/logs/bridge-test.log"
! rg -q "^FAIL" "$G1_DEPLOY_HOME/logs/bridge-test.log"
python scripts/deploy/sim2sim.py --cmd g2,g3,report \
  --clips shanshen_duosuomao_lean35kfs_pad2 --tag local_lean35kfs --ckpt e24_0355000 --workers 6 --imu-model --waist-coupling --waist-roll-range 0.72 --waist-roll-damping 1.8
python scripts/deploy/sim2sim.py --cmd standby \
  --clips shanshen_duosuomao_lean35kfs_pad2 --tag local_lean35kfs --ckpt e24_0355000 --workers 6 --imu-model --waist-coupling --waist-roll-range 0.72 --waist-roll-damping 1.8
