"""Render recorded policy-driven physics beside the reference, with playback labels."""
import json
import argparse
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / 'SuperMimic-Repro-Sim2Real'
sys.path.insert(0, str(REPO / 'tools/video2motion'))
import g1_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clip', default='IMG_3307_arms_stand')
    ap.add_argument('--alias', default='arms')
    ap.add_argument('--model', default='e11c_0220000')
    ap.add_argument('--latency-ms', type=int, default=0)
    ap.add_argument('--work-dir', type=Path, default=ROOT / 'video_motion/work/skill')
    a = ap.parse_args()
    work = a.work_dir
    trace = np.load(work / f'{a.model}_lat{a.latency_ms}.npz', allow_pickle=False)
    raw = np.load(REPO / f'deploy_artifacts/clips/{a.clip}.npz', allow_pickle=False)
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'g1_deploy/unitree_mujoco/unitree_robots/g1/scene_29dof.xml'))
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(640, model.vis.global_.offheight)
    actual, ref = mujoco.MjData(model), mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 640, 640)
    cfg = json.loads((REPO / f'deploy_artifacts/{a.model}/deploy_config.json').read_text())
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in cfg['joint_names']]
    phase = trace['phase']
    moving = np.flatnonzero(phase == 'policy')
    if len(trace['t']) == 0:
        raise ValueError('No policy-driven states were recorded')
    start, end = (trace['t'][moving[0]], trace['t'][moving[-1]]) if len(moving) else (trace['t'][0], trace['t'][-1])
    selected = np.flatnonzero((trace['t'] >= start-2) & (trace['t'] <= end+3))[::2]
    # Deploy observations are heading-relative. Align reference heading to simulation start.
    q0 = raw['root_quat_wxyz'][0]
    yaw = np.arctan2(2*(q0[0]*q0[3]+q0[1]*q0[2]), 1-2*(q0[2]**2+q0[3]**2))
    rz = Rotation.from_euler('z', -yaw)
    reference_origin = raw['root_pos'][0].copy(); reference_origin[2] = 0
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 2.7, -12, 180
    camera.lookat[:] = [0, 0, .80]
    out = work / f'{a.alias}_policy_vs_reference.webm'
    # VP8 plays with this desktop's installed codecs.
    with imageio.get_writer(out, fps=25, codec='libvpx', quality=8, macro_block_size=2,
                            ffmpeg_params=['-b:v','2M','-deadline','good','-cpu-used','4']) as writer:
        for count, i in enumerate(selected):
            frame = int(trace['frame'][i]) if phase[i] == 'policy' else (len(raw['joint_pos'])-1 if phase[i] == 'standby_post' else 0)
            actual.qpos[:] = trace['qpos'][i]
            mujoco.mj_forward(model, actual)
            ref.qpos[:3] = rz.apply(raw['root_pos'][frame]-reference_origin)
            xyzw = (rz * Rotation.from_quat(raw['root_quat_wxyz'][frame][[1,2,3,0]])).as_quat()
            ref.qpos[3:7] = xyzw[[3,0,1,2]]
            ref.qpos[qadr] = raw['joint_pos'][frame]
            mujoco.mj_forward(model, ref)
            panels = []
            for data, title in ((actual, f'{a.model} - physics simulation'), (ref, 'Reference motion')):
                renderer.update_scene(data, camera)
                panel = renderer.render().copy()
                cv2.rectangle(panel, (0,0), (640,72), (25,25,25), -1)
                cv2.putText(panel, title, (15,30), 0, .62, (240,240,240), 2)
                cv2.putText(panel, f"{str(phase[i])} | {trace['t'][i]-start:+.2f} s", (15,57), 0, .5, (215,215,215), 1)
                panels.append(panel)
            joined = np.concatenate(panels, axis=1)
            writer.append_data(joined)
            if count in (90,125,150):
                imageio.imwrite(work / f'{a.alias}_sim_frame_{count}.jpg', joined)
    renderer.close()
    print('Rendered',out,len(selected),'frames',flush=True)


if __name__ == '__main__':
    main()
