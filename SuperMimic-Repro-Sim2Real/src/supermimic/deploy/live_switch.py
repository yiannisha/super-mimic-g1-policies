"""Preloaded motion bank and continuous policy handoff; no DDS or hardware access."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from .core import Clip, DeployConfig, DeployCore, OnnxPolicy, StepResult, static_clip


def require_compatible(source: DeployConfig, target: DeployConfig) -> None:
    if source.joint_names != target.joint_names:
        raise ValueError('policy bank joint ordering differs')
    for field in ('default_joint_pos', 'action_scale', 'kp', 'kd', 'effort_limit',
                  'joint_lower', 'joint_upper', 'velocity_limit', 'soft_limit_factor', 'control_dt'):
        if not np.allclose(getattr(source, field), getattr(target, field), rtol=0, atol=1e-9):
            raise ValueError(f'policy bank {field} differs; live switching requires identical robot constants')


@dataclass(frozen=True)
class Motion:
    alias: str
    model: str
    clip: Clip
    policy: object


class PolicyBank:
    def __init__(self, manifest: Path, root: Path, cfg: DeployConfig, threads: int = 1):
        entries = json.loads(manifest.read_text())
        if not isinstance(entries, dict) or not entries:
            raise ValueError('policy bank must be a nonempty JSON object')
        self.motions = {}
        self.clip_paths = {}
        policies = {}
        for alias, spec in entries.items():
            if not isinstance(alias, str) or not alias or len(alias) > 80:
                raise ValueError('invalid motion alias')
            if not isinstance(spec, dict):
                raise ValueError('each motion must specify model and clip')
            for key in ('model', 'clip'):
                if not isinstance(spec.get(key), str) or not spec[key] or Path(spec[key]).name != spec[key] or spec[key] in ('.', '..'):
                    raise ValueError(f'invalid {key} in policy bank')
            model = spec['model']
            target_cfg = DeployConfig.load(root / model / 'deploy_config.json')
            require_compatible(cfg, target_cfg)
            path = root / 'clips' / (spec['clip'] + '.npz')
            self.clip_paths[alias] = path
            clip = Clip.load(path)
            with np.load(path, allow_pickle=False) as raw:
                if raw['joint_names'].tolist() != cfg.joint_names:
                    raise ValueError(f'{alias}: clip joint ordering differs')
                if not np.allclose(raw['default_joint_pos'], cfg.default_joint_pos, rtol=0, atol=1e-6):
                    raise ValueError(f'{alias}: clip default pose differs')
                if not np.isclose(float(np.ravel(raw['fps'])[0]), 1 / cfg.control_dt):
                    raise ValueError(f'{alias}: clip frame rate differs')
            if (clip.num_frames < 2 or clip.goal.shape != (clip.num_frames, 68)
                    or clip.joint_pos.shape != (clip.num_frames, 29)
                    or clip.proj_grav.shape != (clip.num_frames, 3)
                    or not all(np.isfinite(x).all() for x in (clip.goal, clip.joint_pos, clip.proj_grav))):
                raise ValueError(f'{alias}: invalid motion table')
            if model not in policies:
                policy = OnnxPolicy(root / model / 'policy.onnx', threads=threads)
                # All model allocation/initialization happens before the DDS control loop.
                for _ in range(3):
                    action = policy(np.zeros(930, np.float32), np.zeros(680, np.float32))
                    if action.shape != (29,) or not np.isfinite(action).all():
                        raise ValueError(f'{model}: invalid warm-up output')
                policies[model] = policy
            self.motions[alias] = Motion(alias, model, clip, policies[model])


class PolicyHandoff:
    """Move a standing reference and crossfade policies without clearing history.

    Both policies see the executed blended action in their history. Existing
    DeployCore aborts remain active. This is a handoff mechanism, not a proof of
    balance for an arbitrary pair of policies or poses.
    """
    def __init__(self, source: DeployCore, target: DeployCore, seconds: float, same_policy: bool = False):
        require_compatible(source.cfg, target.cfg)
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('policy handoff duration must be positive and finite')
        self.source, self.target = source, target
        self.first = static_clip(source.clip, source.frame)
        self.last = static_clip(target.clip, 0)
        ga, gb = self.first.proj_grav[0], self.last.proj_grav[0]
        cosine = np.dot(ga, gb) / (np.linalg.norm(ga) * np.linalg.norm(gb))
        if not np.isfinite(cosine) or np.arccos(np.clip(cosine, -1, 1)) > source.safety.max_tilt_err:
            raise ValueError('standing reference orientations differ beyond the existing tilt limit')
        self.steps = max(2, int(np.ceil(seconds / source.cfg.control_dt)) + 1)
        phase = np.linspace(0., 1., self.steps)
        self.weights = .5 - .5 * np.cos(np.pi * phase)
        duration = (self.steps - 1) * source.cfg.control_dt
        rates = (.5 * np.pi / duration * np.sin(np.pi * phase))[:, None]
        weights = self.weights[:, None]
        goals = (1 - weights) * self.first.goal[0] + weights * self.last.goal[0]
        grav = (1 - weights) * ga + weights * gb
        norm = np.linalg.norm(grav, axis=1, keepdims=True)
        grav /= norm
        dgrav = ((gb - ga) * rates) / norm
        dgrav -= grav * np.sum(grav * dgrav, axis=1, keepdims=True)
        goals[:, 58:61] = grav
        goals[:, 29:58] = (self.last.joint_pos[0] - self.first.joint_pos[0]) * rates
        goals[:, 61:64] = -np.cross(grav, dgrav)
        goals[:, 64:67] = 0
        goals[:, 66] = (self.last.goal[0, 67] - self.first.goal[0, 67]) * rates[:, 0]
        joints = (1 - weights) * self.first.joint_pos[0] + weights * self.last.joint_pos[0]
        # A complete reference preserves the normal ten-frame lookahead. A
        # sequence of changing static goals would hide the coming pose change.
        self.reference = Clip('policy_handoff', goals.astype(np.float32), joints, grav)
        self.tick = 0
        self.done = False
        self.same_policy = same_policy
        target.hist = None if source.hist is None else source.hist.copy()
        target.last_action = source.last_action.copy()

    def step(self, q, dq, gyro, quat) -> StepResult:
        if self.done:
            raise RuntimeError('policy handoff already completed')
        weight = self.weights[self.tick]
        self.source.switch_clip(self.reference, self.tick)
        self.target.switch_clip(self.reference, self.tick)
        old = self.source.step(q, dq, gyro, quat, advance=False)
        if self.same_policy:
            new = old
            self.target.hist = self.source.hist.copy()
        else:
            new = self.target.step(q, dq, gyro, quat, advance=False)
        action = (1 - weight) * old.action + weight * new.action
        q_target, n_clamped = self.target.targets(action, np.asarray(q), np.asarray(dq))
        self.source.last_action = action.copy()
        self.target.last_action = action.copy()
        self.tick += 1
        self.done = self.tick == self.steps
        if self.done:
            self.target.switch_clip(self.last, 0)
        return StepResult(q_target, action, 0, self.done, old.abort or new.abort, n_clamped)
