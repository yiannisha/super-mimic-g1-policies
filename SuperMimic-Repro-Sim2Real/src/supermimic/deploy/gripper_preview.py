"""MuJoCo reference playback for gripper event authoring; never connects to DDS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from supermimic.deploy.grippers import GripperEvents, sidecar_path


@dataclass
class SkillPreview:
    path: Path
    fps: float
    joint_names: list[str]
    joint_pos: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    events: GripperEvents
    saved_events: GripperEvents

    @classmethod
    def load(cls, path: str | Path) -> SkillPreview:
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            fps = float(np.asarray(z["fps"]).reshape(-1)[0])
            names = z["joint_names"].tolist()
            joints = z["joint_pos"].astype(float)
            root = z["root_pos"].astype(float)
            quat = z["root_quat_wxyz"].astype(float)
        if joints.ndim != 2 or joints.shape != (len(joints), len(names)) or not len(joints):
            raise ValueError(f"{path}: invalid joint positions")
        if root.shape != (len(joints), 3) or quat.shape != (len(joints), 4):
            raise ValueError(f"{path}: missing or invalid root poses")
        if (
            not np.isfinite(fps)
            or fps <= 0
            or not all(np.isfinite(x).all() for x in (joints, root, quat))
            or not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-3)
        ):
            raise ValueError(f"{path}: invalid frame rate or non-finite/non-unit poses")
        events = GripperEvents.load(sidecar_path(path), len(joints))
        return cls(path, fps, names, joints, root, quat, events, events)

    @property
    def num_frames(self) -> int:
        return len(self.joint_pos)

    @property
    def dirty(self) -> bool:
        return self.events != self.saved_events

    def save(self) -> None:
        self.events.save(sidecar_path(self.path))
        self.saved_events = self.events

    def reload(self) -> None:
        events = GripperEvents.load(sidecar_path(self.path), self.num_frames)
        self.events = self.saved_events = events


class ReferencePreview:
    """Set exact reference poses, with schematic wrist markers for binary gripper targets.

    No physics step, policy inference or physical gripper actuation happens here.
    This lets the editor seek in either direction and inspect an exact clip frame.
    """

    def __init__(self, scene: str | Path):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.motion = None
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Gripper preview needs a G1 model with one free root joint")
        self.root_adr = int(self.model.jnt_qposadr[free[0]])
        self.wrists = [self.model.body(f"{h}_wrist_yaw_link").id for h in ("left", "right")]

    def select(self, motion: SkillPreview) -> None:
        addresses = []
        for name in motion.joint_names:
            joint = self.model.joint(name)
            if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"{name} is not a hinge joint")
            addresses.append(int(joint.qposadr[0]))
        self.joint_adr = np.asarray(addresses)
        self.motion = motion
        self.seek(0)

    def seek(self, frame: int) -> None:
        motion = self.motion
        if motion is None or not 0 <= frame < motion.num_frames:
            raise ValueError("Frame is outside the selected skill")
        d, q = self.data, self.root_adr
        mujoco.mj_resetData(self.model, d)
        d.qpos[q : q + 3] = motion.root_pos[frame]
        d.qpos[q + 3 : q + 7] = motion.root_quat[frame]
        d.qpos[self.joint_adr] = motion.joint_pos[frame]
        d.time = frame / motion.fps
        mujoco.mj_forward(self.model, d)

    def draw_markers(self, scene: mujoco.MjvScene, frame: int) -> None:
        """Visible labels and colored spheres above both hands; no collision geometry."""
        scene.ngeom = 0
        for hand, body, state in zip(("L", "R"), self.wrists, self.motion.events.state_at(frame)):
            if scene.ngeom >= scene.maxgeom:
                break
            transform = self.data.xmat[body].reshape(3, 3)
            pos = self.data.xpos[body] + transform @ np.array([0.11, 0.0, 0.09])
            color = [0.1, 0.9, 0.3, 1.0] if state == "open" else [1.0, 0.35, 0.05, 1.0]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, 0.025),
                pos,
                np.eye(3).ravel(),
                np.array(color, dtype=np.float32),
            )
            geom.label = f"{hand} {state.upper()}"
            scene.ngeom += 1
