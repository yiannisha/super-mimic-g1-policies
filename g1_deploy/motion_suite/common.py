"""CPU-only reference preparation using the pinned upstream converter and robot XML."""
from pathlib import Path
import json
import sys
import numpy as np

HOME = Path('/home/yiannis/policies/g1_deploy')
REPO = HOME.parent / 'SuperMimic-Repro-Sim2Real'
XML = HOME / 'mjlab-assets/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
sys.path.insert(0, str(REPO / 'src'))
from supermimic.data import motion_convert as mc


def model():
    return mc.load_g1_model(XML)


def inverse(q, v):
    q = np.asarray(q, np.float32)
    v = np.asarray(v, np.float32)
    qw, xyz = q[:, :1], q[:, 1:]
    # Same quaternion inverse rotation as mjlab's quat_apply_inverse.
    return v - 2 * qw * np.cross(xyz, v) + 2 * np.cross(xyz, np.cross(xyz, v))


def features(jp, jv, pos, quat, lv, av, q0):
    jp, jv, pos, quat, lv, av, q0 = [np.asarray(x, np.float32) for x in (jp, jv, pos, quat, lv, av, q0)]
    grav = inverse(quat, np.broadcast_to(np.array([0, 0, -1], np.float32), (len(quat), 3)))
    w, x, y, z = quat.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    h = np.zeros_like(quat)
    h[:, 0], h[:, 3] = np.cos(yaw / 2), np.sin(yaw / 2)
    goal = np.concatenate((jp - q0, jv, grav, inverse(quat, av), inverse(h, lv), pos[:, 2:3]), axis=1)
    return goal.astype(np.float32), grav


def export(source, destination, config):
    z = dict(np.load(source, allow_pickle=False))
    cfg = json.loads(Path(config).read_text())
    names = z['joint_names'].tolist()
    assert names == cfg['joint_names'], 'Joint order mismatch'
    assert np.isclose(float(np.ravel(z['fps'])[0]), 50), 'References must be 50 Hz'
    root = z['body_names'].tolist().index('pelvis')
    jp, jv = z['joint_pos'], z['joint_vel']
    pos, quat = z['body_pos_w'][:, root], z['body_quat_w'][:, root]
    lv, av = z['body_lin_vel_w'][:, root], z['body_ang_vel_w'][:, root]
    q0 = np.array(cfg['default_joint_pos'], np.float32)
    goal, grav = features(jp, jv, pos, quat, lv, av, q0)
    assert goal.shape == (len(jp), 68) and np.isfinite(goal).all()
    extra = {k: z[k] for k in ('segment_names', 'segment_kind', 'segment_start', 'segment_end') if k in z}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, name=np.array(destination.stem), fps=np.float32(50), joint_names=np.array(names),
             default_joint_pos=q0, goal=goal, joint_pos=jp.astype(np.float32), proj_grav=grav,
             root_pos=pos, root_quat_wxyz=quat, root_lin_vel_w=lv, root_ang_vel_w=av,
             joint_vel=jv, locked_joints=np.array([], dtype=np.int64), **extra)
    return {'file': str(destination), 'frames': len(jp), 'duration_s': (len(jp)-1)/50}
