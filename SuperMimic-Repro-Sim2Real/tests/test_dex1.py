"""DEX1 target timing, worker guards and runner integration; no hardware or DDS."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from supermimic.deploy.dex1 import ARM, DISARM, PACKET, STOP, Dex1Config, Dex1Guard, worker
from supermimic.deploy.grippers import GripperEvents, GripperPlayback, sidecar_path


def packet(now, mode=ARM, left=0, right=0):
    return PACKET.pack(now, mode, left, right)


def test_playback_uses_original_frames_for_replay_handoff_and_last_frame(tmp_path):
    path = tmp_path / "pickup.npz"
    events = GripperEvents.from_dict({"right": [[2, "closed"], [4, "open"], [7, "closed"]]}, 8)
    events.save(sidecar_path(path))
    playback = GripperPlayback([path, tmp_path / "wave.npz"], [8, 20])
    assert playback.enabled
    assert playback.resolve("PASSIVE", 0) is None
    assert playback.resolve("STANDBY_PRE", 0) == ("open", "open")
    assert playback.resolve("POLICY", 0, 1) == ("open", "open")
    assert playback.resolve("POLICY", 0, 2) == ("open", "closed")
    assert playback.resolve("END_HOLD", 0) == ("open", "closed")
    assert playback.resolve("POLICY", 0, 4, freeze=True) == ("open", "closed")
    assert playback.resolve("POLICY", 0, 4) == ("open", "open")
    assert playback.resolve("STANDBY_POST", 0) == ("open", "closed")
    assert playback.frame == 7
    # Handoff's synthetic frame zero must not prematurely reset the gripper.
    assert playback.resolve("STANDBY_POST", 0, 0, freeze=True) == ("open", "closed")
    assert playback.resolve("STANDBY_PRE", 0) == ("open", "open")
    assert playback.resolve("STANDBY_PRE", 1) is None
    assert playback.resolve("DAMP", 0) is None


def test_guard_binary_goals_and_feedback_displacement_limit():
    guard = Dex1Guard(Dex1Config())
    assert guard.commands(10, (2.7, 2.7), (10, 10)) is None
    guard.receive(packet(10, right=1), 10)
    assert guard.commands(10, (2.7, 2.7), (10, 10)) == pytest.approx([2.88, 2.52])
    assert guard.commands(10, (5.4, 0), (10, 10)) == pytest.approx([5.4, 0])
    guard.receive(packet(10.02, DISARM), 10.02)
    assert guard.commands(10.02, (5.4, 0), (10, 10)) is None
    guard.receive(packet(10.04, STOP), 10.04)
    guard.receive(packet(10.06, ARM), 10.06)
    assert guard.commands(10.06, (5.4, 0), (10, 10)) is None


@pytest.mark.parametrize("fault", ["heartbeat", "feedback", "nan"])
def test_guard_fault_latches_and_cannot_rearm(fault):
    guard = Dex1Guard(Dex1Config())
    guard.receive(packet(10), 10)
    now, q, times = 10.05, (1.0, 2.0), (10.0, 10.0)
    if fault == "heartbeat":
        now = 10.11
    elif fault == "feedback":
        times = (9.0, 10.0)
    else:
        q = (float("nan"), 2.0)
    with pytest.raises(RuntimeError):
        guard.commands(now, q, times)
    guard.receive(packet(now), now)
    assert guard.commands(now, (1.0, 2.0), (now, now)) is None


@pytest.mark.parametrize(
    "data", [b"", packet(0), packet(float("nan")), packet(10, 99), packet(10, left=2)]
)
def test_rejects_stale_or_invalid_parent_packets(data):
    with pytest.raises((ValueError, RuntimeError)):
        Dex1Guard(Dex1Config()).receive(data, 10)


@pytest.mark.parametrize(
    "kwargs", [{"open_q": float("nan")}, {"closed_q": 6.0}, {"kp": 500.0}, {"heartbeat_s": 0}]
)
def test_rejects_invalid_calibration_and_gains(kwargs):
    with pytest.raises(ValueError):
        Dex1Config(**kwargs)


def test_independent_encoder_offsets_and_calibration_file(tmp_path):
    path = tmp_path / "grippers.json"
    path.write_text(
        json.dumps(
            {
                "left": {"open": 5.6003336906433105, "closed": 0.26161277294158936},
                "right": {"open": 10.500213623046875, "closed": 5.120029449462891},
            }
        )
    )
    cfg = Dex1Config.calibrated(path)
    assert cfg.open_q == (5.6003336906433105, 10.500213623046875)
    assert cfg.closed_q == (0.26161277294158936, 5.120029449462891)
    # The worker receives calibration over JSON, so it must accept pairs decoded as lists.
    from dataclasses import asdict

    assert Dex1Config(**json.loads(json.dumps(asdict(cfg)))) == cfg
    guard = Dex1Guard(cfg)
    guard.receive(packet(10, right=1), 10)
    assert guard.commands(10, cfg.open_q, (10, 10)) == pytest.approx(
        [cfg.open_q[0], cfg.open_q[1] - 0.18]
    )
    assert guard.commands(10, (cfg.open_q[0], cfg.closed_q[1]), (10, 10)) == pytest.approx(
        [cfg.open_q[0], cfg.closed_q[1]]
    )
    assert Dex1Config.calibrated(path, open_q=5.5, closed_q=0).open_q == (5.5, 5.5)


def test_wrong_encoder_offset_is_refused_before_driving():
    guard = Dex1Guard(Dex1Config())
    guard.receive(packet(10, right=1), 10)
    with pytest.raises(RuntimeError, match="right encoder.*calibrated range"):
        guard.commands(10, (5.4, 10.5), (10, 10))
    assert guard.stopped


@pytest.mark.parametrize(
    "data",
    [
        {"left": {"open": 5.4, "closed": 0}},
        {"left": {"open": 5.4, "closed": 0}, "right": {"open": 10.5}},
        {"left": {"open": 5.4, "closed": 0}, "right": {"open": 10.5, "closed": 0}},
    ],
)
def test_bad_calibration_profile_rejected(tmp_path, data):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        Dex1Config.calibrated(path)


def test_worker_parent_exit_releases_effort_without_open_command():
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    read_fd, write_fd = os.pipe()
    commands = []

    class Transport:
        def feedback(self):
            return (2.7, 2.7), (time.monotonic(), time.monotonic())

        def publish(self, positions, config):
            commands.append((positions, config))

    errors = []

    def run():
        try:
            worker(child, Transport(), Dex1Config(), parent_fd=read_fd)
        except Exception as exc:  # noqa: BLE001 - return worker-thread failures to the test
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        parent.settimeout(2)
        assert json.loads(parent.recv(4096))["ready"]
        assert not commands  # discovery/feedback alone must never move grippers
        parent.send(packet(time.monotonic(), right=1))
        deadline = time.monotonic() + 0.5
        while not commands and time.monotonic() < deadline:
            time.sleep(0.001)
        assert commands and commands[0][0] == pytest.approx([2.88, 2.52])
        os.close(write_fd)
        write_fd = None
        thread.join(1)
        assert not thread.is_alive() and not errors
        assert commands[-1] == ((2.7, 2.7), None)  # kp/kd=0 at measured aperture, no forced open
    finally:
        if write_fd is not None:
            os.close(write_fd)
        thread.join(1)
        os.close(read_fd)
        parent.close()
        child.close()


def test_runner_frames_and_abort_through_real_loop_without_dds(tmp_path):
    import numpy as np

    repo = Path(__file__).resolve().parents[1]
    artifacts = repo / "deploy_artifacts/e11c_0220000"
    if not (artifacts / "policy.onnx").exists():
        pytest.skip("local deploy bundle not available")
    clip = tmp_path / "pickup.npz"
    goal = np.zeros((8, 68), np.float32)
    goal[:, 60] = -1
    np.savez(
        clip,
        name="pickup",
        goal=goal,
        joint_pos=np.zeros((8, 29)),
        proj_grav=np.tile([0.0, 0.0, -1.0], (8, 1)),
    )
    GripperEvents.from_dict({"right": [[2, "closed"], [4, "open"], [7, "closed"]]}, 8).save(
        sidecar_path(clip)
    )
    receipt = tmp_path / "receipt.json"
    code = f"""
import json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, {str(repo / "scripts/deploy/bridge")!r})
import g1_runner_bridge as mod

def startup(self):
    self.state = 'POLICY'
    self.state_t = time.monotonic()
    self.core = mod.DeployCore(self.cfg, lambda obs, goal: np.zeros(29), self.clips[0], self.safety)
    self.core.reset(0)
    runner = self
    class Worker:
        def __init__(self): self.sent = []; self.stops = 0
        def check(self): pass
        def send(self, states):
            self.sent.append([runner.gripper_playback.frame, states])
            if len(self.sent) == 9: runner.stop.set()
        def stop(self): self.stops += 1
        def close(self): pass
    self.gripper_worker = Worker()

def run(self):
    self.read_state = lambda: SimpleNamespace(q=self.cfg.default_joint_pos.copy(), dq=np.zeros(29),
        gyro=np.zeros(3), quat=np.array([1.,0.,0.,0.]), tick=1, temp=np.full((29,2),25.),
        motorstate=np.zeros(29,dtype=np.int64))
    self.state_age = lambda *args: 0.
    self.send = lambda *args, **kwargs: True
    original_run(self)
    sent = self.gripper_worker.sent.copy()
    self.damp('test stop')
    before = len(self.gripper_worker.sent)
    self.gripper_tick('POLICY', 0, 2)
    assert len(self.gripper_worker.sent) == before
    assert self.gripper_worker.stops >= 1
    Path({str(receipt)!r}).write_text(json.dumps(sent))

original_run = mod.Runner.run
mod.Runner.startup = startup
mod.Runner.run = run
mod.main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--iface",
            "lo",
            "--domain",
            "1",
            "--sim",
            "--stage",
            "ground",
            "--artifacts",
            str(artifacts),
            "--clip",
            str(clip),
            "--log-dir",
            str(tmp_path / "logs"),
            "--cmd-log-max-s",
            "1",
            "--rec-max-s",
            "1",
            "--status-s",
            "100000",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(receipt.read_text())
    assert [row[0] for row in rows] == [0, 1, 2, 3, 4, 5, 6, 7, 7]
    assert [row[1][1] for row in rows] == [
        "open",
        "open",
        "closed",
        "closed",
        "open",
        "open",
        "open",
        "closed",
        "closed",
    ]
