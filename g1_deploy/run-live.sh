#!/usr/bin/env bash
set -euo pipefail
if (( $# == 0 )); then set -- idle; fi
exec /home/yiannis/policies/g1_deploy/run-motion.sh "$@" --live
