"""Tune per-skill open/closed gripper frames in a MuJoCo reference viewer.

  python scripts/deploy/tune_grippers.py --skill pickup

Uses the deploy Python environment (MuJoCo, numpy, Tk); no training stack or DDS.
The controls window edits <clip>.grippers.json next to each exported motion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from supermimic.deploy.grippers import HANDS, sidecar_path


def skill_paths(clips_dir: Path, bank: Path | None) -> dict[str, Path]:
    """Expose friendly skill aliases, plus any exported clips absent from the bank."""
    paths = {}
    if bank is not None:
        for alias, entry in json.loads(bank.read_text()).items():
            path = clips_dir / (entry["clip"] + ".npz")
            if path.is_file():
                paths[alias] = path
    known = set(paths.values())
    for path in sorted(clips_dir.glob("*.npz")):
        if path not in known:
            alias = path.stem
            if alias in paths:
                alias = path.name
            paths[alias] = path
    return paths


class GripperTuner:
    def __init__(self, root, paths, skill, scene):
        import tkinter as tk
        from tkinter import ttk

        from supermimic.deploy.gripper_preview import ReferencePreview

        self.root, self.paths = root, paths
        self.preview = ReferencePreview(scene)
        self.sessions = {}
        self.current = None
        self.frame = 0
        self.playing = False
        self.running = True
        self.keys = queue.SimpleQueue()
        self.last_state = None
        self.last_time = time.monotonic()
        self.accumulator = 0.0
        self.refreshing = False
        root.title("Gripper timing — skill controls")
        root.geometry("650x850")
        root.protocol("WM_DELETE_WINDOW", self.close)
        panel = ttk.Frame(root, padding=12)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="Gripper timing", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(panel, text="Reference preview · green = OPEN · orange = CLOSED").pack(anchor="w")
        skill_row = ttk.Frame(panel)
        skill_row.pack(fill="x", pady=(12, 4))
        ttk.Label(skill_row, text="Skill").pack(side="left", padx=(0, 8))
        self.skill = tk.StringVar(value=skill)
        skill_menu = ttk.Combobox(
            skill_row, textvariable=self.skill, values=list(paths), state="readonly"
        )
        skill_menu.pack(side="left", fill="x", expand=True)
        skill_menu.bind("<<ComboboxSelected>>", lambda _: self.select_skill(self.skill.get()))
        self.info = tk.StringVar()
        ttk.Label(panel, textvariable=self.info, wraplength=610).pack(anchor="w")
        self.frame_text = tk.StringVar()
        ttk.Label(panel, textvariable=self.frame_text, font=("TkDefaultFont", 13, "bold")).pack(
            anchor="w", pady=8
        )
        self.timeline = ttk.Scale(panel, from_=0, to=1, command=self.scrub)
        self.timeline.pack(fill="x")
        transport = ttk.Frame(panel)
        transport.pack(fill="x", pady=5)
        self.play_button = ttk.Button(transport, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left")
        for label, action in (
            ("Start", lambda: self.seek(0)),
            ("◀ 1", lambda: self.seek(self.frame - 1)),
            ("1 ▶", lambda: self.seek(self.frame + 1)),
        ):
            ttk.Button(transport, text=label, command=action, width=7).pack(side="left", padx=2)
        self.speed = tk.StringVar(value="0.5")
        ttk.Combobox(
            transport,
            textvariable=self.speed,
            values=("0.1", "0.25", "0.5", "1", "2"),
            width=5,
            state="readonly",
        ).pack(side="left", padx=6)
        ttk.Label(transport, text="× speed").pack(side="left")
        self.loop = tk.BooleanVar(value=False)
        ttk.Checkbutton(transport, text="Loop", variable=self.loop).pack(side="right")
        goto = ttk.Frame(panel)
        goto.pack(fill="x", pady=4)
        ttk.Label(goto, text="Go to frame").pack(side="left")
        self.goto_frame = tk.StringVar(value="0")
        self.goto_spin = ttk.Spinbox(goto, from_=0, to=1, textvariable=self.goto_frame, width=8)
        self.goto_spin.pack(side="left", padx=6)
        self.goto_spin.bind("<Return>", lambda _: self.go_to_frame())
        ttk.Button(goto, text="Go", command=self.go_to_frame).pack(side="left")
        self.state_text = tk.StringVar()
        ttk.Label(panel, textvariable=self.state_text, font=("TkDefaultFont", 12, "bold")).pack(
            anchor="w", pady=6
        )
        quick = ttk.LabelFrame(panel, text="Set an event at the displayed frame", padding=6)
        quick.pack(fill="x", pady=5)
        for hand in HANDS:
            for state in ("open", "closed"):
                ttk.Button(
                    quick,
                    text=f"{hand.title()} {state}",
                    command=lambda h=hand, s=state: self.mark(h, s),
                ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Label(panel, text="Events hold until the next event. Frames start at 0.").pack(
            anchor="w", pady=(8, 2)
        )
        self.events = ttk.Treeview(
            panel,
            columns=("hand", "frame", "time", "state"),
            show="headings",
            height=7,
            selectmode="browse",
        )
        for column, width in (("hand", 90), ("frame", 80), ("time", 90), ("state", 110)):
            self.events.heading(
                column,
                text={"hand": "Hand", "frame": "Frame", "time": "Time (s)", "state": "State"}[
                    column
                ],
            )
            self.events.column(column, width=width)
        self.events.pack(fill="both", expand=True)
        self.events.bind("<<TreeviewSelect>>", self.choose_event)
        self.events.bind("<Double-1>", lambda _: self.go_to_event())
        edit = ttk.Frame(panel)
        edit.pack(fill="x", pady=5)
        self.edit_hand = tk.StringVar(value="right")
        self.edit_frame = tk.StringVar(value="0")
        self.edit_state = tk.StringVar(value="closed")
        ttk.Combobox(
            edit, values=HANDS, textvariable=self.edit_hand, state="readonly", width=8
        ).pack(side="left")
        self.event_spin = ttk.Spinbox(edit, from_=0, to=1, textvariable=self.edit_frame, width=8)
        self.event_spin.pack(side="left", padx=5)
        ttk.Combobox(
            edit, values=("open", "closed"), textvariable=self.edit_state, state="readonly", width=8
        ).pack(side="left")
        ttk.Button(edit, text="Add / replace", command=self.add_event).pack(side="left", padx=5)
        ttk.Button(edit, text="Update selected", command=lambda: self.add_event(move=True)).pack(
            side="left"
        )
        event_actions = ttk.Frame(panel)
        event_actions.pack(fill="x")
        ttk.Button(
            event_actions,
            text="Use current frame",
            command=lambda: self.edit_frame.set(str(self.frame)),
        ).pack(side="left")
        ttk.Button(event_actions, text="Go to selected event", command=self.go_to_event).pack(
            side="left", padx=5
        )
        ttk.Button(event_actions, text="Delete selected", command=self.delete_event).pack(
            side="left"
        )
        persistence = ttk.Frame(panel)
        persistence.pack(fill="x", pady=(12, 4))
        ttk.Button(persistence, text="Save this skill", command=self.save).pack(side="left")
        ttk.Button(persistence, text="Reload JSON", command=self.reload).pack(side="left", padx=6)
        self.file_text = tk.StringVar()
        ttk.Label(panel, textvariable=self.file_text, wraplength=610).pack(anchor="w")
        self.status = tk.StringVar()
        ttk.Label(panel, textvariable=self.status, wraplength=610).pack(anchor="w", pady=4)
        self.log = tk.Text(panel, height=4, state="disabled", font=("TkFixedFont", 9))
        self.log.pack(fill="x", pady=4)
        ttk.Label(
            panel,
            text="In the 3D window: P play/pause · ←/→ step · Home restart · S save · R reload",
        ).pack(anchor="w")
        self.select_skill(skill)

    @property
    def motion(self):
        return self.sessions[self.current]

    def note(self, message):
        self.status.set(message)
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        # Keep the panel log bounded on long looping sessions.
        if int(self.log.index("end-1c").split(".")[0]) > 200:
            self.log.delete("1.0", "2.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def error(self, error):
        from tkinter import messagebox

        messagebox.showerror("Gripper timing", str(error), parent=self.root)

    def pause(self):
        self.playing = False
        self.play_button.configure(text="Play")
        self.accumulator = 0.0

    def select_skill(self, name):
        from supermimic.deploy.gripper_preview import SkillPreview

        self.pause()
        try:
            path = self.paths[name]
            # Cache by clip path: aliases of the same motion share edits and dirty state.
            motion = self.sessions.get(path)
            if motion is None:
                motion = SkillPreview.load(path)
            self.preview.select(motion)
        except (ValueError, KeyError, OSError) as exc:
            self.skill.set(self.current_alias if self.current is not None else "")
            self.error(exc)
            return
        self.sessions[path] = motion
        self.current, self.current_alias = path, name
        self.skill.set(name)
        self.info.set(f"{path.stem} · {motion.fps:g} fps · frames 0–{motion.num_frames - 1}")
        self.timeline.configure(to=motion.num_frames - 1)
        self.goto_spin.configure(to=motion.num_frames - 1)
        self.event_spin.configure(to=motion.num_frames - 1)
        self.edit_frame.set("0")
        self.last_state = None
        self.refresh_events()
        self.seek(0)
        self.note(f"Selected {name}. Unsaved edits are kept when switching skills.")

    def refresh_events(self):
        self.events.delete(*self.events.get_children())
        rows = sorted(
            (frame, hand, state)
            for hand in HANDS
            for frame, state in getattr(self.motion.events, hand)
        )
        for frame, hand, state in rows:
            self.events.insert(
                "",
                "end",
                iid=f"{hand}:{frame}",
                values=(hand, frame, f"{frame / self.motion.fps:.3f}", state.upper()),
            )
        self.refresh_status()

    def refresh_status(self):
        left, right = self.motion.events.state_at(self.frame)
        self.frame_text.set(
            f"Frame {self.frame} / {self.motion.num_frames - 1}    |    {self.frame / self.motion.fps:.3f} s"
        )
        self.state_text.set(f"Left: {left.upper()}          Right: {right.upper()}")
        self.file_text.set(
            f"{'UNSAVED · ' if self.motion.dirty else ''}{sidecar_path(self.motion.path)}"
        )
        states = left, right
        if states != self.last_state:
            self.note(
                f"{self.current_alias} · frame {self.frame}: L {left.upper()} / R {right.upper()}"
            )
        self.last_state = states

    def seek(self, frame, pause=True):
        if self.current is None:
            return
        if pause:
            self.pause()
        self.frame = max(0, min(int(frame), self.motion.num_frames - 1))
        self.preview.seek(self.frame)
        self.refreshing = True
        self.timeline.set(self.frame)
        self.refreshing = False
        self.goto_frame.set(str(self.frame))
        self.refresh_status()
        self.last_time = time.monotonic()

    def scrub(self, value):
        frame = round(float(value))
        if not self.refreshing and frame != self.frame:
            self.seek(frame)

    def go_to_frame(self):
        try:
            frame = int(self.goto_frame.get())
            if not 0 <= frame < self.motion.num_frames:
                raise ValueError(f"Frame must be in 0..{self.motion.num_frames - 1}")
            self.seek(frame)
        except ValueError as exc:
            self.error(exc)

    def toggle_play(self):
        if self.current is None:
            return
        if self.playing:
            self.pause()
        else:
            if self.frame == self.motion.num_frames - 1:
                self.seek(0)
            self.playing = True
            self.play_button.configure(text="Pause")
            self.last_time = time.monotonic()

    def choose_event(self, _=None):
        selected = self.events.selection()
        if selected:
            hand, frame, _, state = self.events.item(selected[0], "values")
            self.edit_hand.set(hand)
            self.edit_frame.set(frame)
            self.edit_state.set(state.lower())

    def go_to_event(self):
        selected = self.events.selection()
        if selected:
            self.seek(int(selected[0].split(":")[1]))

    def mark(self, hand, state):
        self.pause()
        self.motion.events = self.motion.events.set_event(
            hand, self.frame, state, self.motion.num_frames
        )
        self.refresh_events()
        self.events.selection_set(f"{hand}:{self.frame}")
        self.choose_event()

    def add_event(self, move=False):
        self.pause()
        try:
            events = self.motion.events
            if move:
                selected = self.events.selection()
                if not selected:
                    raise ValueError("Select an event to update")
                hand, frame = selected[0].split(":")
                events = events.remove_event(hand, int(frame), self.motion.num_frames)
            events = events.set_event(
                self.edit_hand.get(),
                int(self.edit_frame.get()),
                self.edit_state.get(),
                self.motion.num_frames,
            )
            self.motion.events = events
        except ValueError as exc:
            self.error(exc)
            return
        self.refresh_events()

    def delete_event(self):
        self.pause()
        for key in self.events.selection():
            hand, frame = key.split(":")
            self.motion.events = self.motion.events.remove_event(
                hand, int(frame), self.motion.num_frames
            )
        self.refresh_events()

    def save(self):
        if self.current is None:
            return
        try:
            self.motion.save()
            self.refresh_status()
            self.note(f"Saved {self.current_alias} gripper events.")
        except OSError as exc:
            self.error(exc)

    def reload(self):
        from tkinter import messagebox

        if self.current is None:
            return
        self.pause()
        if self.motion.dirty and not messagebox.askyesno(
            "Reload gripper events",
            "Discard this skill's unsaved edits and reload its JSON?",
            parent=self.root,
        ):
            return
        try:
            self.motion.reload()
            self.refresh_events()
            self.note(f"Reloaded {self.current_alias} gripper events.")
        except (ValueError, OSError) as exc:
            self.error(exc)

    def close(self):
        from tkinter import messagebox

        dirty = [p.stem for p, session in self.sessions.items() if session.dirty]
        if dirty:
            choice = messagebox.askyesnocancel(
                "Unsaved gripper events",
                "Save edits before closing?\n\n" + "\n".join(dirty),
                parent=self.root,
            )
            if choice is None:
                return
            if choice:
                try:
                    for session in self.sessions.values():
                        if session.dirty:
                            session.save()
                except OSError as exc:
                    self.error(exc)
                    return
        self.running = False

    def tick(self):
        """Run in the Tk/main thread; the MuJoCo key callback only queues commands."""
        import glfw

        actions = {
            glfw.KEY_P: self.toggle_play,
            glfw.KEY_LEFT: lambda: self.seek(self.frame - 1),
            glfw.KEY_RIGHT: lambda: self.seek(self.frame + 1),
            glfw.KEY_HOME: lambda: self.seek(0),
            glfw.KEY_S: self.save,
            glfw.KEY_R: self.reload,
        }
        while not self.keys.empty():
            action = actions.get(self.keys.get())
            if action:
                action()
        now = time.monotonic()
        elapsed, self.last_time = now - self.last_time, now
        if self.playing:
            self.accumulator += min(elapsed, 0.25) * float(self.speed.get()) * self.motion.fps
            steps = int(self.accumulator)
            self.accumulator -= steps
            # Visit every frame, even at high speed, so the event log never skips a transition.
            for _ in range(steps):
                if self.frame == self.motion.num_frames - 1:
                    if self.loop.get():
                        self.seek(0, pause=False)
                    else:
                        self.pause()
                        break
                else:
                    self.seek(self.frame + 1, pause=False)

    def run(self):
        import mujoco
        import mujoco.viewer

        if self.current is None:
            raise ValueError("Could not load the initial skill")
        with mujoco.viewer.launch_passive(
            self.preview.model,
            self.preview.data,
            key_callback=self.keys.put,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            w, x, y, z = self.motion.root_quat[0]
            heading = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 2.8, -15, heading + 180
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SELECTION
            viewer.cam.lookat[:] = self.motion.root_pos[self.frame]
            viewer_closed = False
            while self.running:
                self.root.update()
                if not viewer.is_running():
                    if not viewer_closed:
                        self.pause()
                        self.close()
                        viewer_closed = True
                        if self.running:
                            self.note(
                                "3D window closed. Save your edits and close the controls to exit."
                            )
                    time.sleep(0.03)
                    continue
                with viewer.lock():
                    self.tick()
                    self.preview.draw_markers(viewer.user_scn, self.frame)
                # Overlay setters acquire their own viewer mutex; never nest them in lock().
                if hasattr(viewer, "set_texts"):
                    viewer.set_texts(
                        (
                            None,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            f"{self.current_alias} | reference preview\n{self.frame_text.get()}\n{self.state_text.get()}",
                            "P play/pause | arrows step | S save | R reload",
                        )
                    )
                viewer.sync()
                time.sleep(0.01)
        self.root.destroy()


def main():
    deploy_home = Path(os.environ.get("G1_DEPLOY_HOME", REPO.parent / "g1_deploy"))
    default_bank = deploy_home / "motion_suite/motions.json"
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--skill", help="Skill alias or exported clip name (default: first skill)")
    ap.add_argument("--clips-dir", type=Path, default=REPO / "deploy_artifacts/clips")
    ap.add_argument(
        "--policy-bank",
        type=Path,
        default=default_bank if default_bank.exists() else None,
        help="Optional skill aliases from the deployment motion bank",
    )
    ap.add_argument(
        "--scene",
        type=Path,
        default=Path(
            os.environ.get(
                "G1_UNITREE_XML", deploy_home / "unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
            )
        ),
    )
    args = ap.parse_args()
    try:
        paths = skill_paths(args.clips_dir, args.policy_bank)
        if not paths:
            ap.error(f"No exported clips in {args.clips_dir}")
        skill = args.skill or next(iter(paths))
        if skill not in paths:
            # Also accept a clip stem when it has a friendly bank alias.
            skill = next((key for key, path in paths.items() if path.stem == skill), skill)
        if skill not in paths:
            ap.error(f"Unknown skill {skill!r}; choose one of: {', '.join(paths)}")
        if not args.scene.is_file():
            ap.error(f"Missing G1 scene: {args.scene}; set --scene or source g1_deploy/env.sh")
        import tkinter as tk

        root = tk.Tk()
        GripperTuner(root, paths, skill, args.scene).run()
    except (OSError, ValueError) as exc:
        ap.exit(1, f"Gripper tuner: {exc}\n")


if __name__ == "__main__":
    main()
