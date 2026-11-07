#!/usr/bin/env bash
# Independent videos use independent GPUs; never split a temporal clip.
set -euo pipefail
TASK_ROOT=${V2M_CLOUD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
mkdir -p "$TASK_ROOT/logs"
"$TASK_ROOT/env/bin/python" - "$TASK_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
job = json.loads((root / 'cloud/job.json').read_text())
for name, clip in job['clips'].items():
    actual = hashlib.sha256((root / f'inputs/{name}.mp4').read_bytes()).hexdigest()
    assert actual == clip['input_sha256'], f'{name}: input fingerprint mismatch'
    print(name, 'input verified', actual, flush=True)
PY
run_one() {
    local name=$1 gpu=$2 result=0
    bash "$TASK_ROOT/cloud/run-extraction.sh" "$name" "$gpu" > "$TASK_ROOT/logs/$name.log" 2>&1 || result=$?
    printf '%s\n' "$result" > "$TASK_ROOT/logs/$name.exit"
    return "$result"
}
run_one crunch "${CRUNCH_GPU:-1}" &
crunch_pid=$!
run_one pickup "${PICKUP_GPU:-7}" &
pickup_pid=$!
crunch_result=0
pickup_result=0
wait "$crunch_pid" || crunch_result=$?
wait "$pickup_pid" || pickup_result=$?
printf 'crunch exit=%s; pickup exit=%s\n' "$crunch_result" "$pickup_result"
(( crunch_result == 0 && pickup_result == 0 ))
