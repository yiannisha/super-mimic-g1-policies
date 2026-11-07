"""Download the exact four already-verified public checkpoints on the GPU VM."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root/'cloud/checkpoints.json').read_text())


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(entry):
    out = root/'GVHMR/inputs/checkpoints'/entry['path']
    out.parent.mkdir(parents=True, exist_ok=True)
    size, expected = entry['size'], entry['lfs']['oid']
    if out.exists() and out.stat().st_size == size and checksum(out) == expected:
        print('Verified existing',entry['path'],flush=True)
        return
    part = out.with_name(out.name+'.part')
    for attempt in range(5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset < size:
                url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{quote(entry['path'])}?download=true&ts={int(time.time())}"
                with requests.get(url, headers={'Range':f'bytes={offset}-'} if offset else {},
                                  stream=True,timeout=(30,120)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        offset = 0
                    with part.open('ab' if offset else 'wb') as f:
                        for block in response.iter_content(1 << 20):f.write(block)
            if part.stat().st_size != size or checksum(part) != expected:
                raise RuntimeError('Checkpoint size or SHA-256 mismatch')
            part.replace(out)
            print('Verified',entry['path'],size,'bytes',flush=True)
            return
        except Exception as exc:
            print('Retry',entry['path'],type(exc).__name__,flush=True)
            if attempt == 4:raise
            time.sleep(5)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch,manifest['files']))
