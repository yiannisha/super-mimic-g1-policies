"""Binary gripper events on a clip's zero-based playback timeline.

Sidecars are named ``<clip>.grippers.json``. An event takes effect at its frame
and persists until the next event; a hand is open before its first event.
This module has no simulator, training, or hardware dependencies.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Hand = Literal["left", "right"]
State = Literal["open", "closed"]
HANDS = ("left", "right")
STATES = ("open", "closed")


def sidecar_path(clip_path: str | Path) -> Path:
    return Path(clip_path).with_suffix(".grippers.json")


@dataclass(frozen=True)
class GripperEvents:
    left: tuple[tuple[int, State], ...] = ((0, "open"),)
    right: tuple[tuple[int, State], ...] = ((0, "open"),)

    @classmethod
    def from_dict(cls, payload: dict, num_frames: int) -> GripperEvents:
        if not isinstance(payload, dict) or set(payload) - set(HANDS):
            raise ValueError("Gripper JSON must be an object with only left/right event lists")
        if num_frames < 1:
            raise ValueError("A motion must contain at least one frame")
        tracks = {}
        for hand in HANDS:
            entries = payload.get(hand, [[0, "open"]])
            if not isinstance(entries, list):
                # Treat invalid JSON content consistently as a document validation error.
                raise ValueError(f"{hand}: expected a list of [frame, state] events")  # noqa: TRY004
            track = {}
            for entry in entries:
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise ValueError(f"{hand}: expected [frame, state], got {entry!r}")
                frame, state = entry
                if type(frame) is not int or not 0 <= frame < num_frames:
                    raise ValueError(f"{hand}: frame must be an integer in 0..{num_frames - 1}")
                if state not in STATES:
                    raise ValueError(f"{hand}: state must be open or closed")
                if frame in track:
                    raise ValueError(f"{hand}: duplicate event at frame {frame}")
                track[frame] = state
            tracks[hand] = tuple(sorted(track.items()))
        return cls(**tracks)

    @classmethod
    def load(cls, path: str | Path, num_frames: int) -> GripperEvents:
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()), num_frames)

    def to_dict(self) -> dict:
        return {hand: [list(event) for event in getattr(self, hand)] for hand in HANDS}

    def state_at(self, frame: int) -> tuple[State, State]:
        """Evaluate absolute frame, including seeks backwards, resets and repeated frames."""
        states = []
        for hand in HANDS:
            state = "open"
            for event_frame, event_state in getattr(self, hand):
                if event_frame > frame:
                    break
                state = event_state
            states.append(state)
        return tuple(states)

    def set_event(self, hand: Hand, frame: int, state: State, num_frames: int) -> GripperEvents:
        if hand not in HANDS:
            raise ValueError("Hand must be left or right")
        payload = self.to_dict()
        payload[hand] = [e for e in payload[hand] if e[0] != frame] + [[frame, state]]
        return self.from_dict(payload, num_frames)

    def remove_event(self, hand: Hand, frame: int, num_frames: int) -> GripperEvents:
        if hand not in HANDS:
            raise ValueError("Hand must be left or right")
        payload = self.to_dict()
        payload[hand] = [e for e in payload[hand] if e[0] != frame]
        return self.from_dict(payload, num_frames)

    def save(self, path: str | Path) -> None:
        """Replace the file atomically so a reader never sees partially written JSON."""
        path = Path(path)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
                tmp = Path(f.name)
                json.dump(self.to_dict(), f, indent=2)
                f.write("\n")
            os.replace(tmp, path)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)


class GripperPlayback:
    """Resolve original clip frames; synthetic standby/handoff frames never advance events."""

    def __init__(self, paths: list[Path], lengths: list[int]):
        if len(paths) != len(lengths):
            raise ValueError("Gripper clip paths and lengths must match")
        self.paths = [sidecar_path(p) for p in paths]
        self.lengths = lengths
        # A missing sidecar means leave that skill's grippers alone.
        self.tracks = [
            GripperEvents.load(p, n) if p.is_file() else None for p, n in zip(self.paths, lengths)
        ]
        self.target = None
        self.frame = -1

    @property
    def enabled(self) -> bool:
        return any(track is not None for track in self.tracks)

    def resolve(self, phase: str, index: int, frame: int = 0, freeze: bool = False):
        if phase in ("PASSIVE", "DAMP"):
            self.target, self.frame = None, -1
        elif freeze:
            pass  # preserve the last command through a handoff or controlled thermal stop
        elif self.tracks[index] is None:
            self.target, self.frame = None, -1
        elif phase in ("INTERP", "HOLD", "STANDBY_PRE", "POLICY", "STANDBY_POST"):
            if phase == "STANDBY_POST":
                frame = self.lengths[index] - 1
            elif phase != "POLICY":
                frame = 0
            if not 0 <= frame < self.lengths[index]:
                raise ValueError("Gripper frame is outside the original clip")
            self.target = self.tracks[index].state_at(frame)
            self.frame = frame
        # END_HOLD keeps the previous command, including an exit from pre-standby.
        return self.target
