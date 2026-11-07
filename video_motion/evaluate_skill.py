"""Record the existing Unitree-model standby simulation, using the deploy ONNX core.

No DDS or hardware interfaces. Logging subclasses observe the upstream simulator;
they do not change its policy, integration, gains, abort thresholds, or reference.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / 'SuperMimic-Repro-Sim2Real'
sys.path.insert(0, str(REPO / 'scripts/deploy'))
import sim2sim as sim


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--latency-ms', type=int, default=0)
    ap.add_argument('--model', default='e11c_0220000')
    ap.add_argument('--clip', default='IMG_3307_arms_stand')
    ap.add_argument('--work-dir', type=Path, default=ROOT / 'video_motion/work/skill')
    ap.add_argument('--track-group', choices=['arms', 'legs', 'all'], default='arms')
    a = ap.parse_args()
    work = a.work_dir
    name = a.clip
    preparation = json.loads((work / 'preparation.json').read_text())
    segments = [s for s in preparation['segments'] if s['kind'] == 'motion']
    expected_frames = sum(s['end']-s['start'] for s in segments)
    full_clip_frames = sim.load_clip(name).num_frames
    records = []
    active_robot = []
    moving_started = [False]
    original_robot, original_core = sim.Robot, sim.DeployCore

    class LoggedRobot(original_robot):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            active_robot[:] = [self]

    class LoggedCore(original_core):
        def step(self, q, dq, gyro, quat_wxyz, advance=True):
            if advance:
                moving_started[0] = True
            phase = 'policy' if advance else ('standby_post' if moving_started[0] else 'standby_pre')
            ref_frame = self.frame if advance else (full_clip_frames - 1 if moving_started[0] else 0)
            qref = self.clip.joint_pos[min(self.frame, self.clip.num_frames - 1)].copy()
            robot = active_robot[0]
            result = super().step(q, dq, gyro, quat_wxyz, advance=advance)
            records.append(dict(t=float(robot.d.time), phase=phase, frame=ref_frame,
                qpos=robot.d.qpos.copy(), q=np.array(q), ref=qref, abort=result.abort or ''))
            return result

    sim.Robot, sim.DeployCore = LoggedRobot, LoggedCore
    options = dict(tag='ground', **sim.PROFILES['ground'], ckpt=a.model, clip=name,
                   lat=a.latency_ms, kps=1., mass=0., fric=1., pre_s=10., post_s=10.,
                   r1x_cap_s=10., r1x_win_s=3., r1x_min_s=3.)
    print('Simulating', options, flush=True)
    result = sim.standby_run(options)
    policy = [r for r in records if r['phase'] == 'policy']
    gesture = [r for r in policy if any(s['start'] <= r['frame'] < min(s['end'], full_clip_frames-1) for s in segments)]
    # DeployCore ends after reaching the last frame, so an unpadded clip has N-1 steps.
    expected_steps = sum(max(0, min(s['end'], full_clip_frames-1)-s['start']) for s in segments)
    errors = np.array([r['q'] - r['ref'] for r in gesture])
    if len(errors):
        arm_rmse = float(np.sqrt(np.mean(errors[:, 15:]**2)))
        selected = errors[:, 15:] if a.track_group == 'arms' else (errors[:, :15] if a.track_group == 'legs' else errors)
        selected_rmse = float(np.sqrt(np.mean(selected**2)))
        tracking = dict(gesture_steps=len(errors), all_joint_rmse_rad=float(np.sqrt(np.mean(errors**2))),
                       selected_group=a.track_group, selected_rmse_rad=selected_rmse, expected_steps=expected_steps,
                       arms_rmse_rad=arm_rmse, arms_max_abs_error_rad=float(np.abs(errors[:, 15:]).max()),
                       per_joint_rmse_rad=dict(zip(sim.load_cfg(a.model).joint_names, np.sqrt(np.mean(errors**2, axis=0)).tolist())))
    else:
        arm_rmse = selected_rmse = float('inf')
        tracking = dict(gesture_steps=0, expected_steps=expected_steps)
    # The hardware runner gates playback on a recent settled window, not the
    # maximum since releasing the gantry. Use the last five seconds here as a
    # conservative superset of its three-second window. Preserve the upstream
    # whole-phase pass result for comparison; never change controller limits.
    standby = [result['phases'].get(phase, {}) for phase in ('standby_pre', 'standby_post')]
    settled_tilt = max(p.get('settled_max_tilt_err_deg', float('inf')) for p in standby)
    settled_dq = max(p.get('settled_max_dq', float('inf')) for p in standby)
    standby_ready = bool(settled_tilt < sim.STANDBY_R1X_TILT_DEG and settled_dq < sim.STANDBY_R1X_DQ)
    accepted = bool(result['completed'] and result['upright'] and result['aborts'] == 0
                    and standby_ready and not result['r1x_never'] and result['smooth']
                    and len(gesture) == expected_steps and selected_rmse < 0.20)
    result.update(tracking=tracking, accepted=accepted,
                  standby_settled_max_tilt_err_deg=settled_tilt, standby_settled_max_dq=settled_dq,
                  standby_ready=standby_ready,
                  acceptance_rule=f'Completes with zero falls/aborts; settled standby meets runner tilt/speed limits; smooth handoffs; all {expected_steps} motion steps run; {a.track_group} RMSE <0.20rad',
                  simulation_only=True)
    prefix = work / f'{a.model}_lat{a.latency_ms}'
    prefix.with_suffix('.json').write_text(json.dumps(sim.strip(result), indent=2)+'\n')
    np.savez_compressed(prefix.with_suffix('.npz'), t=np.array([r['t'] for r in records]),
        phase=np.array([r['phase'] for r in records]), frame=np.array([r['frame'] for r in records]),
        qpos=np.array([r['qpos'] for r in records]), q=np.array([r['q'] for r in records]),
        ref=np.array([r['ref'] for r in records]), abort=np.array([r['abort'] for r in records]))
    print(json.dumps(sim.strip(result), indent=2), flush=True)


if __name__ == '__main__':
    main()
