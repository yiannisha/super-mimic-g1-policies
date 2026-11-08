"""Extract a fingerprinted batch using the existing cloud environment, one worker per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import queue
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--gpus', nargs='+', required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    clips = json.loads(a.manifest.read_text())['clips']
    pending = queue.Queue()
    logs = root / 'logs/raw_batch'
    logs.mkdir(parents=True, exist_ok=True)
    for name, entry in clips.items():
        actual = hashlib.sha256((root / f'inputs/{name}.mp4').read_bytes()).hexdigest()
        if actual != entry['normalized_sha256']:
            raise ValueError(f'{name}: input fingerprint mismatch')
        cache = root / f'work/gvhmr/{name}'
        marker = cache / 'input.sha256'
        if cache.exists() and (not marker.exists() or marker.read_text().strip() != actual):
            raise ValueError(f'{name}: stale extraction cache; move it aside before retrying')
        cache.mkdir(parents=True, exist_ok=True)
        marker.write_text(actual + '\n')
        pending.put(name)

    def worker(gpu):
        failed = []
        while True:
            try:
                name = pending.get_nowait()
            except queue.Empty:
                return failed
            print(f'{name}: GPU {gpu} starting', flush=True)
            with (logs / f'{name}.log').open('w') as output:
                result = subprocess.run(['bash', str(root / 'cloud/run-extraction.sh'), name, gpu],
                                        stdout=output, stderr=subprocess.STDOUT)
            (logs / f'{name}.exit').write_text(f'{result.returncode}\n')
            print(f'{name}: exit {result.returncode}', flush=True)
            if result.returncode:
                failed.append(name)

    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        failures = sum(pool.map(worker, a.gpus), [])
    if failures:
        raise SystemExit('Extraction failed: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
