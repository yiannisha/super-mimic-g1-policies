"""Wrap a retargeted motion with a known standing reference and export its goal table."""
import importlib.util
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / 'SuperMimic-Repro-Sim2Real'
sys.path.insert(0, str(ROOT / 'g1_deploy/motion_suite'))
import common
from supermimic.data.motion_convert import ResampledMotion, forward_kinematics, _so3_derivative

spec = importlib.util.spec_from_file_location('combo', REPO / 'scripts/deploy/make_combo_clip.py')
combo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combo)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT / 'video_motion/work/main/npz/IMG_3307.npz')
    ap.add_argument('--name', default='IMG_3307_arms_stand')
    ap.add_argument('--work-dir', type=Path, default=ROOT / 'video_motion/work/skill')
    ap.add_argument('--model', default='e11c_0220000')
    ap.add_argument('--transitions', choices=['standing', 'none', 'auto'], default='standing')
    a = ap.parse_args()
    work = a.work_dir
    work.mkdir(parents=True, exist_ok=True)
    name = a.name
    source = dict(np.load(a.input, allow_pickle=False))
    idle = dict(np.load(REPO / 'data/npz/bones_selected/neutral_idle_loop_001__A081.npz', allow_pickle=False))
    assert source['joint_names'].tolist() == idle['joint_names'].tolist()
    r = source['body_names'].tolist().index('pelvis')
    p = source['body_pos_w'][:, r].astype(float)
    q = source['body_quat_w'][:, r].astype(float)
    jp = source['joint_pos'].astype(float)
    idle_p = idle['body_pos_w'][0, r].astype(float)
    idle_q = idle['body_quat_w'][0, r].astype(float)
    idle_j = idle['joint_pos'][0].astype(float)
    endpoint_report = {}
    for side, index, window in [('start', 0, slice(0, 25)), ('end', -1, slice(-25, None))]:
        gravity = common.inverse(q[[index]], np.array([[0., 0., -1.]]))[0]
        tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
        velocity_p95 = float(np.percentile(np.abs(source['joint_vel'][window]), 95))
        standing = bool(abs(p[index, 2]-idle_p[2]) < .12 and tilt < 20.
                        and float(np.abs(jp[index, [3, 9]]).mean()) < 1.1)
        endpoint_report[side] = dict(pelvis_height_m=float(p[index, 2]), tilt_deg=tilt,
            joint_speed_p95_rad_s=velocity_p95, standing=standing, settled=velocity_p95 < 1.)
    mode = a.transitions
    if mode == 'auto':
        mode = 'none' if all(v['standing'] and v['settled'] for v in endpoint_report.values()) else 'standing'
    # Align motion heading/xy with the standing clip, preserving relative motion.
    yaw_delta = float(combo.yaw_of(idle_q) - combo.yaw_of(q[0]))
    p = (p - np.r_[p[0, :2], 0.0]) @ combo.rot_z(yaw_delta).T + np.r_[idle_p[:2], 0.0]
    q = combo.quat_mul(combo.quat_z(yaw_delta)[None], q)
    u = np.arange(1, 75) / 75.0
    w = .5 - .5 * np.cos(np.pi * u)
    segments = []
    pp, qq, jj = [], [], []

    def add(label, kind, pos, quat, joints):
        start = sum(len(x) for x in pp)
        pp.append(pos); qq.append(quat); jj.append(joints)
        segments.append(dict(name=label, kind=kind, start=start, end=start+len(pos)))

    def hold(label):
        add(label, 'stand', np.tile(idle_p, (51, 1)), np.tile(idle_q, (51, 1)), np.tile(idle_j, (51, 1)))

    if mode == 'standing':
        hold('standing lead-in')
        add('standing to recorded movement', 'transition', idle_p+w[:, None]*(p[0]-idle_p),
            combo.slerp(idle_q, q[0], w), idle_j+w[:, None]*(jp[0]-idle_j))
    add(a.input.stem, 'motion', p, q, jp)
    # The outgoing standing pose keeps the end-of-motion xy/heading, as the runner does.
    end_p = np.r_[p[-1, :2], idle_p[2]]
    end_q = combo.quat_mul(combo.quat_z(float(combo.yaw_of(q[-1])-combo.yaw_of(idle_q))), idle_q)
    if mode == 'standing':
        add('recorded movement to standing', 'transition', p[-1]+w[:, None]*(end_p-p[-1]),
            combo.slerp(q[-1], end_q, w), jp[-1]+w[:, None]*(idle_j-jp[-1]))
        add('standing finish', 'stand', np.tile(end_p, (51, 1)), np.tile(end_q, (51, 1)), np.tile(idle_j, (51, 1)))
    p, q, jp = np.concatenate(pp), np.concatenate(qq), np.concatenate(jj)
    motion = ResampledMotion(fps=50., root_pos=p, root_quat=q, joint_pos=jp,
        root_lin_vel_w=np.gradient(p, .02, axis=0), root_ang_vel_w=_so3_derivative(q, .02),
        joint_vel=np.gradient(jp, .02, axis=0))
    model = common.model()
    bodies = forward_kinematics(model, motion)
    target = work / f'{name}.npz'
    np.savez(target, fps=np.array([50.], np.float32), joint_pos=jp.astype(np.float32),
        joint_vel=motion.joint_vel.astype(np.float32), body_pos_w=bodies.pos_w.astype(np.float32),
        body_quat_w=bodies.quat_w.astype(np.float32), body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
        body_ang_vel_w=bodies.ang_vel_w.astype(np.float32), joint_names=source['joint_names'], body_names=source['body_names'],
        source=np.array(str(a.input)), input_fps=np.array([30.]),
        segment_names=np.array([s['name'] for s in segments]), segment_kind=np.array([s['kind'] for s in segments]),
        segment_start=np.array([s['start'] for s in segments]), segment_end=np.array([s['end'] for s in segments]))
    result = common.export(target, REPO / f'deploy_artifacts/clips/{name}.npz',
                           REPO / f'deploy_artifacts/{a.model}/deploy_config.json')
    report = dict(**result, training_npz=str(target), segments=segments,
                  original_motion_preserved=True, transitions_requested=a.transitions,
                  transitions_used=mode, endpoints=endpoint_report,
                  max_joint_speed_rad_s=float(np.abs(motion.joint_vel).max()))
    (work / 'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
