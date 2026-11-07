"""Exercise the real DEX1 worker against a fake service on domain 83 / lo only.

No robot interface, serial device or body command topics are used. The fake
grippers immediately follow received motor targets and report their positions.
Run in the deploy environment: python scripts/deploy/bridge/test_dex1.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from supermimic.deploy.dex1 import Dex1Config, Dex1Process


def main():
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

    ChannelFactoryInitialize(83, "lo")
    lock = threading.Lock()
    positions = [5.4, 5.4]
    received = []
    latest = [None, None]
    subscribers, publishers = [], []
    stopped = threading.Event()
    for index, hand in enumerate(("left", "right")):

        def on_command(message, i=index):
            assert len(message.cmds) == 1
            motor = message.cmds[0]
            with lock:
                received.append((i, motor.q, motor.kp, motor.kd))
                latest[i] = motor

        sub = ChannelSubscriber(f"rt/dex1/{hand}/cmd", MotorCmds_)
        sub.Init(on_command, 1)
        subscribers.append(sub)
        pub = ChannelPublisher(f"rt/dex1/{hand}/state", MotorStates_)
        pub.Init()
        publishers.append(pub)

    def feedback():
        while not stopped.is_set():
            with lock:
                for i, command in enumerate(latest):
                    if command is not None and command.kp > 0:
                        positions[i] = command.q
                state = positions.copy()
            for pub, q in zip(publishers, state):
                motor = unitree_go_msg_dds__MotorState_()
                motor.q = q
                message = MotorStates_()
                message.states = [motor]
                pub.Write(message)
            time.sleep(0.005)

    thread = threading.Thread(target=feedback, daemon=True)
    thread.start()
    child = None
    try:
        with tempfile.TemporaryDirectory() as directory:
            child = Dex1Process(83, "lo", Dex1Config(), Path(directory) / "dex1.log")
            assert not received, "worker published before ARM"

            def play(states, seconds):
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    child.send(states)
                    child.check()
                    time.sleep(0.02)

            play(("open", "closed"), 1.0)
            with lock:
                assert positions[0] > 5.3 and positions[1] < 0.05, positions
                assert any(i == 1 and q < 5.3 and kp == 5.0 for i, q, kp, kd in received)
            print("PASS DDS MotorCmds_/MotorStates_: right closes while left stays open")
            play(("open", "open"), 1.0)
            with lock:
                assert min(positions) > 5.3, positions
            print("PASS DDS opening target")
            child.stop()
            play(("closed", "closed"), 0.15)
            with lock:
                assert all(
                    command is not None and command.kp == 0 and command.kd == 0
                    for command in latest
                )
                before = len(received)
            play(("closed", "closed"), 0.15)
            with lock:
                assert not any(kp > 0 for _, _, kp, _ in received[before:])
            print("PASS DAMP/STOP releases effort and cannot re-arm")
            child.close()
            assert child.process.returncode == 0
            child = None
            stopped.set()
            thread.join(1)
            with lock:
                before = len(received)
            try:
                Dex1Process(83, "lo", Dex1Config(), Path(directory) / "missing.log", timeout=0.2)
            except RuntimeError as exc:
                assert "No fresh DEX1" in str(exc), str(exc)
            else:
                raise AssertionError("missing gripper feedback did not fail startup")
            with lock:
                assert len(received) == before, "failed startup sent a motor command"
            print("PASS missing service refuses startup without motor commands")
    finally:
        stopped.set()
        thread.join(1)
        if child is not None:
            child.close()


if __name__ == "__main__":
    main()
