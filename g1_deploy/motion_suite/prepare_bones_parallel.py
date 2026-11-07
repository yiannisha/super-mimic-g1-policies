"""Read the official authorized archive in resumable parallel HTTP ranges.

Only a small range window is fetched ahead of extraction; stop after the selected
requested clips are prepared. No training, simulation, DDS or motor commands.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from huggingface_hub import get_hf_file_metadata, hf_hub_url, get_token
from common import HOME, REPO, model, mc, export

CHUNK=16*1024*1024
WORKERS=8
CACHE=HOME/'downloads/bones-selected-ranges'
CACHE.mkdir(parents=True,exist_ok=True)
LOG=HOME/'logs/bones-range-status.json'


class ArchiveReader:
    def __init__(self):
        self.lock=threading.Lock()
        self.token=get_token()
        if not self.token:
            raise RuntimeError('Hugging Face login is required')
        self.refresh()
        expected={'size':self.size,'etag':self.etag}
        manifest=CACHE/'source.json'
        if manifest.exists() and json.loads(manifest.read_text())!=expected:
            raise RuntimeError('Source archive changed; use a new range cache')
        manifest.write_text(json.dumps(expected,indent=2)+'\n')
        self.count=math.ceil(self.size/CHUNK)
        self.pool=ThreadPoolExecutor(max_workers=WORKERS)
        self.pending={}
        self.index=0
        self.buffer=b''
        self.offset=0
        self.start=time.monotonic()
        self.reuse_prefix()
        self.fill()

    def refresh(self):
        meta=get_hf_file_metadata(hf_hub_url('bones-studio/seed','g1.tar.gz',repo_type='dataset'),token=self.token)
        self.url,self.size,self.etag=meta.location,meta.size,meta.etag

    def path(self,i):
        return CACHE/f'{i:06d}.chunk'

    def reuse_prefix(self):
        # These files came from the same HTTP endpoint and the hash in their
        # filenames is the source ETag. Xet reconstruction files are excluded.
        base=HOME/'downloads/bones-seed-http/.cache/huggingface/download'
        for p in base.glob(f'*.{self.etag}.*.incomplete'):
            complete=p.stat().st_size//CHUNK
            with p.open('rb') as src:
                for i in range(complete):
                    dst=self.path(i)
                    if dst.exists() and dst.stat().st_size==CHUNK:
                        src.seek(CHUNK,1)
                        continue
                    data=src.read(CHUNK)
                    if len(data)!=CHUNK:
                        break
                    tmp=dst.with_suffix('.copying')
                    tmp.write_bytes(data)
                    tmp.replace(dst)
            print(f'Reused {complete*CHUNK/1e6:.0f} MB from the prior HTTP transfer',flush=True)
            break

    def fill(self):
        for i in range(self.index,min(self.index+WORKERS,self.count)):
            if i not in self.pending:
                self.pending[i]=self.pool.submit(self.fetch,i)

    def fetch(self,i):
        start=i*CHUNK
        end=min(self.size,start+CHUNK)-1
        expected=end-start+1
        dst=self.path(i)
        if dst.exists() and dst.stat().st_size==expected:
            return dst
        tmp=dst.with_suffix('.part')
        for attempt in range(1,21):
            try:
                with self.lock:
                    url=self.url
                req=Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urlopen(req,timeout=90) as r:
                    cr=r.headers.get('Content-Range','')
                    if r.status!=206 or not cr.startswith(f'bytes {start}-{end}/'):
                        raise RuntimeError('Server did not return the requested archive range')
                    with tmp.open('wb') as out:
                        left=expected
                        while left:
                            block=r.read(min(1024*1024,left))
                            if not block:
                                raise IOError('Range response ended early')
                            out.write(block)
                            left-=len(block)
                tmp.replace(dst)
                return dst
            except Exception as exc:
                if attempt==20:
                    raise RuntimeError(f'Archive range {i} failed after 20 attempts ({type(exc).__name__})') from None
                print(f'Range {i}: retry {attempt}/20 after {type(exc).__name__}',flush=True)
                if isinstance(exc,HTTPError) and exc.code in (401,403):
                    with self.lock:
                        self.refresh()
                time.sleep(min(2*attempt,15))

    def read(self,n):
        blocks=[]
        length=0
        while length<n:
            if not self.buffer:
                if self.index>=self.count:
                    break
                p=self.pending.pop(self.index).result()
                self.buffer=p.read_bytes()
                self.index+=1
                self.fill()
                elapsed=time.monotonic()-self.start
                status={'source_bytes_consumed':self.offset,'source_size':self.size,
                        'elapsed_s':elapsed,'ranges_consumed':self.index,'range_size':CHUNK}
                tmp=LOG.with_suffix('.json.tmp')
                tmp.write_text(json.dumps(status,indent=2)+'\n')
                tmp.replace(LOG)
                if self.index%8==0:
                    print(f'Archive read {self.offset/1e9:.2f}/{self.size/1e9:.2f} GB in {elapsed:.0f} s',flush=True)
            take=min(n-length,len(self.buffer))
            blocks.append(self.buffer[:take])
            self.buffer=self.buffer[take:]
            self.offset+=take
            length+=take
        return b''.join(blocks)

    def close(self):
        self.pool.shutdown(wait=False,cancel_futures=True)


def main():
    entries=json.loads((HOME/'motion_suite/motions.json').read_text())
    names={e['clip'] for e in entries.values() if e['model']=='e11c_0220000'}
    raw=REPO/'data/raw/bones_selected'
    out=REPO/'data/npz/bones_selected'
    raw.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    cfg=REPO/'deploy_artifacts/e11c_0220000/deploy_config.json'
    m=model()
    results=[]

    def prepare(name):
        mc.convert_csv_to_npz(raw/(name+'.csv'),out/(name+'.npz'),120,model=m)
        info=export(out/(name+'.npz'),REPO/'deploy_artifacts/clips'/(name+'.npz'),cfg)
        results.append(info)
        print('PREPARED '+name+' '+json.dumps(info),flush=True)

    needed=set(names)
    for name in sorted(names):
        if (raw/(name+'.csv')).exists():
            prepare(name)
            needed.remove(name)
    if needed:
        reader=ArchiveReader()
        try:
            with tarfile.open(fileobj=reader,mode='r|gz',bufsize=1024*1024) as archive:
                folder=''
                for member in archive:
                    if not member.isfile() or not member.name.endswith('.csv'):
                        continue
                    p=Path(member.name)
                    if p.parent.name!=folder:
                        folder=p.parent.name
                        print(f'Scanning {folder} at {reader.offset/1e9:.3f} compressed GB',flush=True)
                    matches=needed & {p.stem,p.stem+'__'+p.parent.name,p.parent.name+'__'+p.stem}
                    if not matches:
                        continue
                    name=matches.pop()
                    dest=raw/(name+'.csv')
                    temp=dest.with_suffix('.csv.part')
                    with archive.extractfile(member) as src,temp.open('wb') as target:
                        shutil.copyfileobj(src,target)
                    temp.replace(dest)
                    prepare(name)
                    needed.remove(name)
                    if not needed:
                        break
        finally:
            reader.close()
    if needed:
        raise RuntimeError('Requested clips not found: '+', '.join(sorted(needed)))
    (HOME/'logs/bones-preparation.json').write_text(json.dumps(results,indent=2)+'\n')
    from launcher import MOTIONS,readiness
    for key,entry in MOTIONS.items():
        ready,detail=readiness(entry)
        print(f"{key}: {'READY' if ready else 'PENDING'} — {detail}",flush=True)
    print('ALL SELECTED DIGITAL MOTION SELECTIONS PREPARED',flush=True)
    subprocess.run([sys.executable,str(HOME/'motion_suite/launcher.py'),'list'],check=True)
    (HOME/'logs/motion-suite-complete.json').write_text(json.dumps({
        'prepared':list(MOTIONS),'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'training_performed':False,'simulation_suite_run':False,'robot_controller_started':False,
    },indent=2)+'\n')


if __name__=='__main__':
    main()
