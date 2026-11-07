"""Inference-only timing of preloaded handoffs. No physics, DDS or robot commands."""
import json
import sys
import time
from pathlib import Path
import numpy as np
HOME=Path(__file__).resolve().parents[1]
ROOT=HOME.parent/'SuperMimic-Repro-Sim2Real/deploy_artifacts'
sys.path.insert(0,str(ROOT.parent/'src'))
from supermimic.deploy.core import DeployConfig,DeployCore,static_clip
from supermimic.deploy.live_switch import PolicyBank,PolicyHandoff
cfg=DeployConfig.load(ROOT/'e11c_0220000/deploy_config.json')
bank=PolicyBank(HOME/'motion_suite/motions.json',ROOT,cfg)
report=[]
for source in ('idle','dance46','dance22'):
    for target in ('idle','dance46','dance22'):
        old,new=bank.motions[source],bank.motions[target]
        with np.load(ROOT/'clips'/(old.clip.name+'.npz')) as raw:
            quat=raw['root_quat_wxyz'][-1]
        q=old.clip.joint_pos[-1]; dq=np.zeros(29); gyro=np.zeros(3)
        before=DeployCore(cfg,old.policy,static_clip(old.clip,old.clip.num_frames-1))
        before.step(q,dq,gyro,quat,advance=False)
        after=DeployCore(cfg,new.policy,static_clip(new.clip,0))
        handoff=PolicyHandoff(before,after,2,same_policy=old.policy is new.policy)
        lat=[]; aborts=set()
        while not handoff.done:
            t=time.perf_counter()
            result=handoff.step(q,dq,gyro,quat)
            lat.append((time.perf_counter()-t)*1000)
            if result.abort:
                aborts.add(result.abort)
            assert np.isfinite(result.q_target).all()
        row={'from':source,'to':target,'ticks':len(lat),'p50_ms':float(np.percentile(lat,50)),
             'p99_ms':float(np.percentile(lat,99)),'max_ms':max(lat),'aborts':sorted(aborts)}
        report.append(row)
        print(json.dumps(row),flush=True)
(HOME/'logs/live-switch-timing.json').write_text(json.dumps(report,indent=2)+'\n')
assert not any(row['aborts'] for row in report)
