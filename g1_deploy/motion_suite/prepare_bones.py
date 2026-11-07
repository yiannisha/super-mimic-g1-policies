"""Download authorized BONES-SEED data and prepare only the selected motions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from common import HOME, REPO, mc, model, export

# A dedicated cache avoids treating Xet's partially reconstructed file as an
# HTTP prefix. Each HTTP retry resumes its own contiguous partial download.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,help='Use an existing authorized g1.tar.gz instead of downloading')
    a=ap.parse_args()
    motions=json.loads((HOME/'motion_suite/motions.json').read_text())
    selected={entry['clip'] for entry in motions.values() if entry['model']=='e11c_0220000'}
    raw=REPO/'data/raw/bones_selected'
    out=REPO/'data/npz/bones_selected'
    raw.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    needed={name for name in selected if not (raw/(name+'.csv')).exists()}
    if needed:
        archive=a.archive
        if archive is None:
            from huggingface_hub import hf_hub_download, get_token
            if not get_token():
                raise SystemExit('BONES-SEED access required: accept the dataset terms, then run hf auth login locally.')
            from huggingface_hub.errors import GatedRepoError
            for attempt in range(1, 21):
                try:
                    archive=Path(hf_hub_download(repo_id='bones-studio/seed',repo_type='dataset',filename='g1.tar.gz',
                                                local_dir=HOME/'downloads/bones-seed-http'))
                    break
                except GatedRepoError:
                    raise SystemExit('Hugging Face denied dataset access. Accept access at https://huggingface.co/datasets/bones-studio/seed using the logged-in account.')
                except Exception as exc:
                    if attempt == 20:
                        raise
                    print(f'HTTP download interrupted ({type(exc).__name__}); resuming attempt {attempt+1}/20 in 10 s', flush=True)
                    time.sleep(10)
        print(f'Extracting {len(needed)} selected CSVs from {archive}',flush=True)
        t0=time.monotonic()
        last=t0
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                if member.isfile() and member.name.lower().endswith('.csv'):
                    path=Path(member.name)
                    candidates={path.stem,path.stem+'__'+path.parent.name,path.parent.name+'__'+path.stem}
                    matches=needed & candidates
                    if len(matches)>1:
                        raise ValueError(f'Ambiguous archive member {member.name}')
                    if matches:
                        name=matches.pop()
                        target=raw/(name+'.csv')
                        tmp=target.with_suffix('.csv.part')
                        src=tar.extractfile(member)
                        with tmp.open('wb') as dst:
                            shutil.copyfileobj(src,dst)
                        tmp.replace(target)
                        needed.remove(name)
                        print(f'Extracted {name}; {len(needed)} remaining',flush=True)
                        if not needed:
                            break
                now=time.monotonic()
                if now-last>30:
                    print(f'Scanning archive: {now-t0:.0f} s, {len(needed)} clips remaining',flush=True)
                    last=now
        if needed:
            raise SystemExit('Archive did not contain requested CSVs: '+', '.join(sorted(needed)))
    m=model()
    cfg=REPO/'deploy_artifacts/e11c_0220000/deploy_config.json'
    results=[]
    for name in sorted(selected):
        info=mc.convert_csv_to_npz(raw/(name+'.csv'),out/(name+'.npz'),120,model=m)
        print(json.dumps(info),flush=True)
        results.append(export(out/(name+'.npz'),REPO/'deploy_artifacts/clips'/(name+'.npz'),cfg))
    (HOME/'logs/bones-preparation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
