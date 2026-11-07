"""Fetch the four public checkpoints from the mirror used by GVHMR's linked demo.

Body models remain user-provided. Check byte counts and SHA-256 from LFS metadata.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
import requests

root = Path(__file__).resolve().parent
repo_id = "camenduru/GVHMR"
info = requests.get(f"https://huggingface.co/api/models/{repo_id}", timeout=30).json()
revision = info["sha"]
entries = requests.get(f"https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true", timeout=30).json()
groups = {"gvhmr", "hmr2", "vitpose", "yolo"}
selected = [e for e in entries if e["type"] == "file" and e["path"].split("/")[0] in groups]
assert len(selected) == 4
(root / "work/public_weights_mirror.json").write_text(json.dumps(dict(repo=repo_id, revision=revision, files=selected), indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(entry):
    dest = root / "GVHMR/inputs/checkpoints" / entry["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected, size = entry["lfs"]["oid"], entry["size"]
    if dest.is_file() and dest.stat().st_size == size and digest(dest) == expected:
        print(f"Verified existing: {entry['path']}", flush=True)
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(6):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset < size:
                url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{quote(entry['path'])}?download=true&ts={int(time.time())}"
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as r:
                    r.raise_for_status()
                    if offset and r.status_code != 206:
                        offset = 0
                    print(f"Downloading {entry['path']} from byte {offset} / {size}", flush=True)
                    with part.open("ab" if offset else "wb") as out:
                        for chunk in r.iter_content(1 << 20):
                            out.write(chunk)
            if part.stat().st_size != size:
                raise ValueError(f"Size {part.stat().st_size} != {size}")
            if digest(part) != expected:
                raise ValueError("SHA-256 mismatch")
            part.replace(dest)
            print(f"Verified {entry['path']}: {size:,} bytes, SHA-256 {expected}", flush=True)
            return
        except Exception as exc:
            print(f"Retry {attempt + 1} {entry['path']}: {type(exc).__name__}", flush=True)
            if attempt == 5:
                raise
            time.sleep(min(5 * (attempt + 1), 30))


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download, selected))
print("All four public checkpoints verified.", flush=True)
