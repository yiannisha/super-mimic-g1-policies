"""Exact reference poses and GUI authoring with synthetic motions; never uses DDS."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from supermimic.deploy.gripper_preview import ReferencePreview, SkillPreview
from supermimic.deploy.grippers import sidecar_path


@pytest.fixture
def assets(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("""<mujoco><worldbody><body name="pelvis">
      <freejoint/><geom type="sphere" size=".1" mass="1"/>
      <body name="left_wrist_yaw_link" pos="0 .3 0">
        <joint name="left_wrist_yaw_joint" axis="0 0 1"/>
        <geom type="sphere" size=".05" mass=".1"/>
      </body>
      <body name="right_wrist_yaw_link" pos="0 -.3 0">
        <joint name="right_wrist_yaw_joint" axis="0 0 1"/>
        <geom type="sphere" size=".05" mass=".1"/>
      </body>
    </body></worldbody></mujoco>""")
    paths = {}
    for name in ("pickup", "wave"):
        path = tmp_path / f"{name}.npz"
        np.savez(
            path,
            fps=50.0,
            joint_names=["left_wrist_yaw_joint", "right_wrist_yaw_joint"],
            joint_pos=np.arange(8).reshape(4, 2) / 10,
            root_pos=np.array([[i, 0, 1] for i in range(4)]),
            root_quat_wxyz=np.tile([1, 0, 0, 0], (4, 1)),
        )
        paths[name] = path
    return scene, paths


def test_preview_seeks_to_exact_pose_and_markers_follow_event_boundary(assets):
    scene_path, paths = assets
    preview = ReferencePreview(scene_path)
    motion = SkillPreview.load(paths["pickup"])
    motion.events = motion.events.set_event("right", 2, "closed", 4)
    preview.select(motion)
    scene = mujoco.MjvScene(preview.model, maxgeom=2)
    for frame in (1, 2, 3, 0):
        preview.seek(frame)
        preview.draw_markers(scene, frame)
        np.testing.assert_allclose(preview.data.qpos[:3], motion.root_pos[frame])
        np.testing.assert_allclose(preview.data.qpos[preview.joint_adr], motion.joint_pos[frame])
        assert preview.data.time == frame / 50
        assert scene.ngeom == 2
        assert scene.geoms[0].label == "L OPEN"
        assert scene.geoms[1].label == ("R CLOSED" if frame >= 2 else "R OPEN")
        np.testing.assert_allclose(
            scene.geoms[1].rgba, [1, 0.35, 0.05, 1] if frame >= 2 else [0.1, 0.9, 0.3, 1]
        )


def test_failed_reload_preserves_unsaved_work(assets):
    _, paths = assets
    motion = SkillPreview.load(paths["pickup"])
    motion.events = motion.events.set_event("right", 2, "closed", 4)
    edited = motion.events
    sidecar_path(motion.path).write_text('{"right": [[999, "closed"]]}')
    with pytest.raises(ValueError):
        motion.reload()
    assert motion.events == edited and motion.dirty
    motion.save()
    assert not motion.dirty
    assert SkillPreview.load(motion.path).events == edited


def tuner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/deploy/tune_grippers.py"
    spec = importlib.util.spec_from_file_location("tune_grippers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_aliases_and_unlisted_clips(assets, tmp_path):
    _, paths = assets
    bank = tmp_path / "motions.json"
    bank.write_text(json.dumps({"grab": {"clip": "pickup"}, "missing": {"clip": "absent"}}))
    found = tuner_module().skill_paths(tmp_path, bank)
    assert found == {"grab": paths["pickup"], "wave": paths["wave"]}


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Tk requires a display")
def test_gui_edit_seek_switch_save_reload_and_playback(assets, monkeypatch):
    import tkinter as tk

    scene, paths = assets
    module = tuner_module()
    root = tk.Tk()
    root.withdraw()
    app = module.GripperTuner(root, paths, "pickup", scene)
    errors = []
    app.error = errors.append
    try:
        app.seek(2)
        app.mark("right", "closed")
        app.select_skill("wave")
        assert app.motion.events.state_at(2) == ("open", "open")
        app.select_skill("pickup")
        app.seek(2)
        assert app.motion.dirty and "CLOSED" in app.state_text.get()
        app.events.selection_set("right:2")
        app.choose_event()
        app.edit_frame.set("3")
        app.add_event(move=True)
        assert app.motion.events.state_at(2) == ("open", "open")
        assert app.motion.events.state_at(3) == ("open", "closed")
        # Invalid moves must not delete the selected event or corrupt the timeline.
        app.events.selection_set("right:3")
        app.edit_frame.set("100")
        app.add_event(move=True)
        assert len(errors) == 1
        assert app.motion.events.state_at(3) == ("open", "closed")
        app.save()
        app.reload()
        app.seek(0)
        app.speed.set("1")
        app.toggle_play()
        app.last_time = 0
        monkeypatch.setattr(module.time, "monotonic", lambda: 0.06)
        app.tick()
        assert app.frame == 3 and app.motion.events.state_at(app.frame) == ("open", "closed")
        # Programmatic timeline updates must not recurse or pause playback.
        root.update()
        assert app.frame == 3 and app.playing
        app.seek(2)
        assert "Right: OPEN" in app.state_text.get()
        app.select_skill("wave")
        assert not sidecar_path(paths["wave"]).exists()
        assert SkillPreview.load(paths["pickup"]).events.state_at(3) == ("open", "closed")
    finally:
        root.destroy()
