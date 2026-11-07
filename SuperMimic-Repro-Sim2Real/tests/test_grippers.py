"""Gripper command boundaries, validation and durable per-clip sidecars (no hardware)."""

import json

import pytest

from supermimic.deploy.grippers import GripperEvents, sidecar_path


def test_events_hold_at_exact_boundaries_and_follow_seeks():
    events = GripperEvents.from_dict(
        {"right": [[120, "closed"], [300, "open"]], "left": [[200, "closed"]]}, 400
    )
    assert events.state_at(119) == ("open", "open")
    assert events.state_at(120) == ("open", "closed")
    assert events.state_at(299) == ("closed", "closed")
    assert events.state_at(300) == ("closed", "open")
    # Seeking backwards and holding a paused frame must not retain a future state.
    assert events.state_at(119) == ("open", "open")
    assert events.state_at(120) == events.state_at(120) == ("open", "closed")
    assert events.state_at(0) == ("open", "open")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"rigth": []},
        {"right": "closed"},
        {"right": [[-1, "open"]]},
        {"right": [[400, "open"]]},
        {"right": [[1.5, "closed"]]},
        {"right": [[True, "closed"]]},
        {"right": [[2, "half"]]},
        {"right": [[2, "open"], [2, "closed"]]},
        {"right": [[2]]},
    ],
)
def test_rejects_ambiguous_or_out_of_range_events(payload):
    with pytest.raises(ValueError):
        GripperEvents.from_dict(payload, 400)


def test_sidecars_are_independent_and_edits_survive_save_reload(tmp_path):
    pickup, wave = (sidecar_path(tmp_path / name) for name in ("pickup.npz", "wave.npz"))
    events = GripperEvents.load(pickup, 10).set_event("right", 0, "closed", 10)
    events = events.set_event("right", 9, "open", 10)
    events.save(pickup)
    assert GripperEvents.load(pickup, 10) == events
    assert GripperEvents.load(wave, 10).state_at(0) == ("open", "open")
    assert json.loads(pickup.read_text())["right"] == [[0, "closed"], [9, "open"]]
    # Replacing an event keeps only one command at that frame.
    events = events.set_event("right", 9, "closed", 10)
    events.save(pickup)
    assert GripperEvents.load(pickup, 10).right == ((0, "closed"), (9, "closed"))
    assert set(tmp_path.iterdir()) == {pickup}
    # An empty track has an implicit open state.
    events = events.remove_event("right", 0, 10).remove_event("right", 9, 10)
    assert events.state_at(9) == ("open", "open")
