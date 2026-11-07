"""Prepare selected CSVs while the resumable HTTP archive is still downloading."""
import json
from pathlib import Path
import shutil
import tarfile
import time
from common import HOME, REPO, model, mc, export

cache=HOME/'downloads/bones-seed-http'
final=cache/'g1.tar.gz'
raw=REPO/'data/raw/bones_selected'
out=REPO/'data/npz/bones_selected'
raw.mkdir(parents=True,exist_ok=True)
out.mkdir(parents=True,exist_ok=True)
config=REPO/'deploy_artifacts/e11c_0220000/deploy_config.json'
entries=json.loads((HOME/'motion_suite/motions.json').read_text())
names={e['clip'] for e in entries.values() if e['model']=='e11c_0220000'}
needed={n for n in names if not (REPO/'deploy_artifacts/clips'/(n+'.npz')).exists()}


def prepare(name):
    info=mc.convert_csv_to_npz(raw/(name+'.csv'),out/(name+'.npz'),120,model=m)
    info=export(out/(name+'.npz'),REPO/'deploy_artifacts/clips'/(name+'.npz'),config)
    print('PREPARED '+name+' '+json.dumps(info),flush=True)


class GrowingReader:
    def __init__(self,path):
        self.f=path.open('rb')
        self.last=time.monotonic()
    def read(self,n):
        data=[]
        total=0
        while total<n:
            part=self.f.read(n-total)
            if part:
                data.append(part)
                total+=len(part)
            elif final.exists():
                break
            else:
                now=time.monotonic()
                if now-self.last>=30:
                    print(f'Waiting for archive bytes at {self.f.tell()/1e9:.3f} GB; {len(needed)} selected clips remaining',flush=True)
                    self.last=now
                time.sleep(.2)
        return b''.join(data)


m=model()
for name in sorted(needed.copy()):
    if (raw/(name+'.csv')).exists():
        prepare(name)
        needed.remove(name)
while needed:
    candidates=list((cache/'.cache/huggingface/download').glob('*.incomplete'))
    if final.exists():
        path=final
    elif candidates:
        path=max(candidates,key=lambda p:p.stat().st_mtime)
    else:
        time.sleep(1)
        continue
    reader=GrowingReader(path)
    folder=''
    with tarfile.open(fileobj=reader,mode='r|gz',bufsize=1024*1024) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.csv'):
                continue
            p=Path(member.name)
            if p.parent.name!=folder:
                folder=p.parent.name
                print(f'Scanning folder {folder} at {reader.f.tell()/1e9:.3f} compressed GB',flush=True)
            matches=needed & {p.stem,p.stem+'__'+p.parent.name,p.parent.name+'__'+p.stem}
            if matches:
                name=matches.pop()
                target=raw/(name+'.csv')
                tmp=target.with_suffix('.csv.part')
                with archive.extractfile(member) as src,tmp.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
                tmp.replace(target)
                prepare(name)
                needed.remove(name)
                if not needed:
                    break
    if needed:
        raise SystemExit('Not found: '+', '.join(sorted(needed)))
print('ALL SELECTED BONES MOTIONS PREPARED',flush=True)
