"""Select a prepared pretrained model and reference; starts hardware only on explicit run."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import numpy as np

HOME = Path(__file__).resolve().parents[1]
REPO = HOME.parent / 'SuperMimic-Repro-Sim2Real'
MOTIONS = json.loads((HOME/'motion_suite/motions.json').read_text())


def readiness(entry):
    artifact = REPO/'deploy_artifacts'/entry['model']
    clip = REPO/'deploy_artifacts/clips'/(entry['clip']+'.npz')
    missing = [str(p) for p in (artifact/'policy.onnx',artifact/'deploy_config.json',clip) if not p.is_file()]
    if missing:
        return False, 'missing: ' + ', '.join(missing)
    config = json.loads((artifact/'deploy_config.json').read_text())
    with np.load(clip, allow_pickle=False) as z:
        assert z['goal'].ndim == 2 and z['goal'].shape[1] == 68
        assert z['joint_pos'].shape == (len(z['goal']),29)
        assert z['joint_names'].tolist() == config['joint_names']
        np.testing.assert_allclose(z['default_joint_pos'], config['default_joint_pos'], atol=1e-6)
        assert abs(float(np.ravel(z['fps'])[0])-50)<1e-6
        assert all(np.isfinite(z[k]).all() for k in ('goal','joint_pos','proj_grav'))
        return True, f"{len(z['goal'])} frames / {(len(z['goal'])-1)/50:.2f} s"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('motion', choices=['list']+list(MOTIONS))
    ap.add_argument('--iface',default='eth0')
    ap.add_argument('--stage',choices=['air','feet','slack','ground'],default='ground')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--no-grippers',action='store_true',help='Ignore saved gripper events; play the body motion only.')
    calibration=HOME/'gripper_calibration.json'
    ap.add_argument('--gripper-calibration',type=Path,default=calibration if calibration.is_file() else None,
                    help='Per-hand DEX1 open/closed encoder calibration JSON.')
    ap.add_argument('--gripper-open',type=float,default=None,help='Override the open encoder position for both hands.')
    ap.add_argument('--gripper-closed',type=float,default=None,help='Override the closed encoder position for both hands.')
    ap.add_argument('--allow-ssh',action='store_true',
                    help='Allow the runner to start over SSH; keep the remote stop available.')
    ap.add_argument('--live',action='store_true',help='Preload the motion bank; select/replay only in STANDBY_POST.')
    ap.add_argument('--control-socket',type=Path,default=HOME/'control.sock')
    ap.add_argument('--ignore-heating-rate',action='store_true',help='Skip predictive heating-rate checks only.')
    ap.add_argument('--no-temperature-damp',action='store_true',help='Warn instead of runtime temperature DAMP; other thermal checks remain.')
    ap.add_argument('--ignore-temperature-safety',action='store_true',
                    help='Disable all runner temperature checks, warnings, start/switch limits and thermal stops; keep temperature logging.')
    a = ap.parse_args()
    if a.motion == 'list':
        report={}
        for key,entry in MOTIONS.items():
            ready,detail=readiness(entry)
            report[key]={**entry,'prepared':ready,'detail':detail}
            print(f"{key:14s} {'READY' if ready else 'PENDING':7s} {entry['model']:16s} {entry['label']} — {detail}")
        status_path=HOME/'logs/motion-suite-status.json'
        temporary=status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(status_path)
        return
    entry=MOTIONS[a.motion]
    ready,detail=readiness(entry)
    if not ready:
        raise SystemExit(detail+'\nFinish digital preparation before launching this motion.')
    cmd=[sys.executable,str(REPO/'scripts/deploy/bridge/g1_runner_bridge.py'),
         '--iface',a.iface,'--domain','0','--stage',a.stage,
         '--artifacts',str(REPO/'deploy_artifacts'/entry['model']),
         '--clip',str(REPO/'deploy_artifacts/clips'/(entry['clip']+'.npz')),
         '--log-dir',str(REPO/'deploy_logs/real')]
    if a.allow_ssh:
        cmd.append('--allow-ssh')
    if a.no_grippers:
        cmd.append('--no-grippers')
    if a.gripper_calibration is not None:
        cmd += ['--gripper-calibration',str(a.gripper_calibration)]
    if a.gripper_open is not None:
        cmd += ['--gripper-open',str(a.gripper_open)]
    if a.gripper_closed is not None:
        cmd += ['--gripper-closed',str(a.gripper_closed)]
    if a.ignore_heating_rate:
        cmd.append('--ignore-heating-rate')
    if a.no_temperature_damp:
        cmd.append('--no-temperature-damp')
    if a.ignore_temperature_safety:
        cmd.append('--ignore-temperature-safety')
    if a.live:
        cmd += ['--policy-bank',str(HOME/'motion_suite/motions.json'),
                '--policy-bank-root',str(REPO/'deploy_artifacts'),
                '--policy-control-socket',str(a.control_socket)]
    print(f"Selected: {a.motion} — {entry['label']} ({detail})",flush=True)
    print(shlex.join(cmd),flush=True)
    if not a.dry_run:
        os.chdir(REPO)
        os.execv(sys.executable,cmd)


if __name__=='__main__':
    main()
