#!/usr/bin/env bash
set -euo pipefail
source /home/yiannis/policies/g1_deploy/env.sh
cd /home/yiannis/policies/SuperMimic-Repro-Sim2Real
exec python scripts/deploy/bridge/ro_bridge_check.py --iface "${1:-eth0}" --domain 0
