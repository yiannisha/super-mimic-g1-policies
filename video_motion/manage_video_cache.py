"""Invalidate one clip's derived files when the source bytes change."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('work', type=Path)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    source, work = args.source.resolve(), args.work.resolve()
    stem = source.stem
    h = hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    stamp = dict(source=str(source), sha256=h.hexdigest(), size_bytes=source.stat().st_size)
    provenance = work / 'sources' / (stem+'.json')
    previous = json.loads(provenance.read_text()) if provenance.exists() else None
    if args.verify:
        if previous != stamp:
            raise SystemExit('Source changed during processing; rerun to process the replacement')
        print('Source fingerprint verified:', stem, stamp['sha256'], flush=True)
        return
    if previous == stamp:
        print('Reusing cache for identical source:', stem, flush=True)
        return
    derived = [work/'video30'/f'{stem}.mp4', work/'gvhmr'/stem]
    for mode in ['drift','constant']:
        derived += [work/f'csv_{mode}'/f'{stem}.csv', work/f'csv_{mode}'/f'{stem}_gmr.pkl']
    derived += [work/'npz'/f'{stem}.npz', work/'npz'/f'{stem}_conversion.json',
                work/'preview'/f'{stem}.mp4', work/'preview'/f'{stem}.webm']
    for path in derived:
        if path.is_dir(): shutil.rmtree(path)
        elif path.exists(): path.unlink()
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(stamp, indent=2)+'\n')
    print('Cleared stale derived files for:', stem, 'source SHA-256:', stamp['sha256'], flush=True)


if __name__ == '__main__':
    main()
