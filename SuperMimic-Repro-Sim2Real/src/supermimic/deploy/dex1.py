"""Isolated DEX1 DDS publisher for the body runner.

The parent sends binary targets/heartbeats over a private Unix socket. The worker
owns SDK callbacks and the 200 Hz stream, keeping their GIL work out of the body
policy. It never publishes until armed, and releases drive effort on disarm,
stale feedback, a stale parent heartbeat, parent exit, or a latched stop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PACKET = struct.Struct("<dBBB")  # monotonic send time, mode, left closed, right closed
DISARM, ARM, STOP = 0, 1, 2


@dataclass(frozen=True)
class Dex1Config:
    # Unitree's calibrated Dex1_1_Gripper_Controller defaults.
    open_q: float | tuple[float, float] = 5.4
    closed_q: float | tuple[float, float] = 0.0
    kp: float = 5.0
    kd: float = 0.05
    max_step: float = (
        0.18  # maximum requested displacement from feedback, not a third logical state
    )
    heartbeat_s: float = 0.1
    feedback_s: float = 0.25

    def __post_init__(self):
        def pair(value):
            values = value if isinstance(value, (tuple, list)) else (value, value)
            if len(values) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) for v in values
            ):
                raise ValueError("DEX1 endpoints must be numbers or a left/right pair")
            return tuple(float(v) for v in values)

        object.__setattr__(self, "open_q", pair(self.open_q))
        object.__setattr__(self, "closed_q", pair(self.closed_q))
        numeric = (
            *self.open_q,
            *self.closed_q,
            self.kp,
            self.kd,
            self.max_step,
            self.heartbeat_s,
            self.feedback_s,
        )
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("DEX1 configuration must be finite")
        if any(
            not 0 < opened - closed <= 5.6 for opened, closed in zip(self.open_q, self.closed_q)
        ):
            raise ValueError(
                "Each DEX1 hand must have a positive calibrated stroke of at most 5.6 rad"
            )
        if not (0 < self.kp <= 5 and 0 <= self.kd <= 0.2 and 0 < self.max_step <= 0.18):
            raise ValueError("DEX1 gains/step exceed the supported Unitree defaults")
        if self.heartbeat_s <= 0 or self.feedback_s <= 0:
            raise ValueError("DEX1 watchdog timeouts must be positive")

    @classmethod
    def calibrated(cls, path=None, open_q=None, closed_q=None, **kwargs):
        opened, closed = 5.4, 0.0
        if path is not None:
            data = json.loads(Path(path).read_text())
            if not isinstance(data, dict) or set(data) != {"left", "right"}:
                raise ValueError("DEX1 calibration needs left and right objects")
            for hand in ("left", "right"):
                if not isinstance(data[hand], dict) or set(data[hand]) != {"open", "closed"}:
                    raise ValueError(f"DEX1 calibration {hand} needs open and closed positions")
            opened = tuple(data[hand]["open"] for hand in ("left", "right"))
            closed = tuple(data[hand]["closed"] for hand in ("left", "right"))
        return cls(
            open_q=opened if open_q is None else open_q,
            closed_q=closed if closed_q is None else closed_q,
            **kwargs,
        )

    def validate_feedback(self, positions):
        for hand, q, low, high in zip(("left", "right"), positions, self.closed_q, self.open_q):
            if not math.isfinite(q) or not low - 0.3 <= q <= high + 0.3:
                raise RuntimeError(
                    f"DEX1 {hand} encoder {q:g} is outside calibrated range {low:g}..{high:g}; check the calibration profile"
                )


class Dex1Guard:
    """Pure worker state, also exercised without DDS in tests."""

    def __init__(self, config: Dex1Config):
        self.config = config
        self.target = None
        self.heartbeat = None
        self.stopped = False

    def receive(self, packet: bytes, now: float):
        if len(packet) != PACKET.size:
            raise ValueError("Invalid DEX1 heartbeat packet")
        stamp, mode, left, right = PACKET.unpack(packet)
        if not math.isfinite(stamp) or mode not in (DISARM, ARM, STOP) or left > 1 or right > 1:
            raise ValueError("Invalid DEX1 heartbeat values")
        if now - stamp > self.config.heartbeat_s or stamp > now + 0.01:
            raise RuntimeError("DEX1 parent heartbeat expired")
        self.heartbeat = stamp
        if mode == STOP:
            self.stopped = True
        self.target = (left, right) if mode == ARM and not self.stopped else None

    def commands(self, now: float, q: tuple[float, float], feedback_times: tuple[float, float]):
        if self.heartbeat is not None and now - self.heartbeat > self.config.heartbeat_s:
            self.stopped = True
            raise RuntimeError("DEX1 parent heartbeat expired")
        if self.stopped or self.target is None:
            return None
        if any(not math.isfinite(v) for v in q) or any(
            now - t > self.config.feedback_s for t in feedback_times
        ):
            self.stopped = True
            raise RuntimeError("DEX1 feedback missing or stale")
        try:
            self.config.validate_feedback(q)
        except RuntimeError:
            self.stopped = True
            raise
        out = []
        for index, (position, closed) in enumerate(zip(q, self.target)):
            goal = self.config.closed_q[index] if closed else self.config.open_q[index]
            out.append(
                max(position - self.config.max_step, min(position + self.config.max_step, goal))
            )
        return out


class Dex1Process:
    def __init__(
        self, domain: int, iface: str, config: Dex1Config, log_path: Path, timeout: float = 5.0
    ):
        self.lock = threading.Lock()
        self.stopped = False
        self.closed = False
        self.error = None
        self.ready = False
        self.last_status = time.monotonic()
        self.sock, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        env = dict(os.environ)
        # Support both source checkouts and an installed deployment package.
        env["PYTHONPATH"] = (
            str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        )
        args = [
            sys.executable,
            "-m",
            "supermimic.deploy.dex1",
            "--fd",
            str(child.fileno()),
            "--domain",
            str(domain),
            "--iface",
            iface,
            "--config",
            json.dumps(asdict(config)),
            "--wait-s",
            str(timeout),
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("ab") as log:
                self.process = subprocess.Popen(
                    args,
                    pass_fds=(child.fileno(),),
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    env=env,
                    start_new_session=True,
                )
        except BaseException:
            self.sock.close()
            raise
        finally:
            child.close()
        try:
            deadline = time.monotonic() + timeout + 2
            while not self.ready:
                self.check()
                if time.monotonic() > deadline:
                    raise RuntimeError(f"DEX1 worker did not become ready (log {log_path})")
                time.sleep(0.01)
        except BaseException:
            self.close()
            raise

    def check(self):
        while True:
            try:
                message = json.loads(self.sock.recv(4096))
            except BlockingIOError:
                break
            self.last_status = time.monotonic()
            self.error = message.get("error") or self.error
            self.ready = message.get("ready", self.ready)
        if self.error:
            raise RuntimeError(self.error)
        if self.process.poll() is not None:
            raise RuntimeError(f"DEX1 worker exited ({self.process.returncode})")
        if self.ready and time.monotonic() - self.last_status > 1.0:
            raise RuntimeError("DEX1 worker stopped reporting status")

    def send(self, states):
        with self.lock:
            if self.closed:
                return
            mode = STOP if self.stopped else (DISARM if states is None else ARM)
            closed = (0, 0) if states is None else tuple(int(s == "closed") for s in states)
            self.sock.send(PACKET.pack(time.monotonic(), mode, *closed))

    def stop(self):
        with self.lock:
            self.stopped = True
        try:
            self.send(None)
        except OSError:
            pass  # the independent worker heartbeat guard also disarms

    def close(self):
        if self.closed:
            return
        self.stop()
        self.closed = True
        if self.process.stdin:
            self.process.stdin.close()  # EOF tells the worker to release effort and exit
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1)
        self.sock.close()


class SdkTransport:
    """All SDK imports and callbacks stay in the worker process."""

    def __init__(self, domain, iface):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

        ChannelFactoryInitialize(domain, iface)
        self.q, self.times = [0.0, 0.0], [float("-inf"), float("-inf")]
        self.lock = threading.Lock()
        self.publishers, self.subscribers, self.messages = [], [], []
        for index, side in enumerate(("left", "right")):
            sub = ChannelSubscriber(f"rt/dex1/{side}/state", MotorStates_)

            def receive(msg, i=index):
                with self.lock:
                    self.q[i] = float(msg.states[0].q) if len(msg.states) == 1 else float("nan")
                    self.times[i] = time.monotonic()

            sub.Init(receive, 1)
            self.subscribers.append(sub)
            pub = ChannelPublisher(f"rt/dex1/{side}/cmd", MotorCmds_)
            pub.Init()
            self.publishers.append(pub)
            msg = MotorCmds_()
            msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
            self.messages.append(msg)

    def feedback(self):
        with self.lock:
            return tuple(self.q), tuple(self.times)

    def publish(self, targets, config):
        for pub, msg, target in zip(self.publishers, self.messages, targets):
            motor = msg.cmds[0]
            motor.q, motor.dq, motor.tau = target, 0.0, 0.0
            motor.kp, motor.kd = (0.0, 0.0) if config is None else (config.kp, config.kd)
            if pub.Write(msg) is False:
                raise RuntimeError("DEX1 command publication failed")


def worker(sock, transport, config, wait_s=5.0, parent_fd=0):
    """Receive the parent, stream at 200 Hz, and report faults without blocking on logs."""
    sock.setblocking(False)
    guard = Dex1Guard(config)
    drove = False

    def report(**message):
        try:
            sock.send(json.dumps(message).encode())
        except OSError:
            pass

    try:
        deadline = time.monotonic() + wait_s
        while True:
            q, times = transport.feedback()
            now = time.monotonic()
            if all(math.isfinite(v) for v in q) and all(
                now - t <= config.feedback_s for t in times
            ):
                config.validate_feedback(q)
                break
            if now > deadline:
                raise RuntimeError(
                    "No fresh DEX1 left/right state: start the calibrated dex1_1_service before the runner"
                )
            if select.select([parent_fd], [], [], 0)[0] and os.read(parent_fd, 64) == b"":
                return
            time.sleep(0.01)
        report(ready=True)
        status_time = 0.0
        while True:
            start = time.monotonic()
            readable, _, _ = select.select([sock, parent_fd], [], [], 0)
            if parent_fd in readable and os.read(parent_fd, 64) == b"":
                break
            if sock in readable:
                # Do not collapse packets: a STOP packet must remain latched even if followed by ARM.
                for _ in range(64):
                    try:
                        guard.receive(sock.recv(128), time.monotonic())
                    except BlockingIOError:
                        break
            q, times = transport.feedback()
            targets = guard.commands(time.monotonic(), q, times)
            if targets is not None:
                drove = True
                transport.publish(targets, config)
            elif drove:
                transport.publish(tuple(v if math.isfinite(v) else 0.0 for v in q), None)
                drove = False
            if start - status_time >= 0.1:
                report(ready=True)
                status_time = start
            time.sleep(max(0.0, 0.005 - (time.monotonic() - start)))
    except Exception as exc:
        report(error=f"DEX1: {exc}")
        raise
    finally:
        if drove:
            q, _ = transport.feedback()
            try:
                transport.publish(tuple(v if math.isfinite(v) else 0.0 for v in q), None)
            except Exception:  # noqa: BLE001, S110
                pass  # service timeout remains the fallback if DDS itself has failed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fd", type=int, required=True)
    ap.add_argument("--domain", type=int, required=True)
    ap.add_argument("--iface", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--wait-s", type=float, default=5.0)
    args = ap.parse_args()
    sock = socket.socket(fileno=args.fd)
    try:
        config = Dex1Config(**json.loads(args.config))
        worker(sock, SdkTransport(args.domain, args.iface), config, args.wait_s)
    except Exception as exc:
        try:
            sock.send(json.dumps({"error": f"DEX1: {exc}"}).encode())
        except OSError:
            pass
        raise


if __name__ == "__main__":
    main()
