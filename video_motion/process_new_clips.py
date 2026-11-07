"""Wait for extraction, then prepare/test/render the two requested motions.

Uses released ONNX weights only. No training, DDS, robot commands, or bank writes.
Natural video transitions are tried first; synthetic transitions are considered
only after a start/end failure. Each stage verifies the source fingerprint.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
V2M = ROOT / 'video_motion'
STATUS = V2M / 'work/new_clips/status.json'
status = json.loads(STATUS.read_text())
env = os.environ.copy()
env.update(G1_DEPLOY_HOME=str(ROOT / 'g1_deploy'),
           G1_UNITREE_XML=str(ROOT / 'g1_deploy/unitree_mujoco/unitree_robots/g1/scene_29dof.xml'),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
env['LD_LIBRARY_PATH'] = str(ROOT / 'g1_deploy/cyclonedds/lib')+':'+str(ROOT / 'g1_deploy/venv/lib')
env.pop('PYTHONPATH', None)
deploy_python = ROOT / 'g1_deploy/venv/bin/python'
render_python = V2M / 'gmr_env/bin/python'


def save():
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, indent=2)+'\n')
    temporary.replace(STATUS)


def verify(alias):
    expected = status['clips'][alias]['sha256']
    actual = hashlib.sha256((ROOT / (alias+'.mp4')).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'{alias}: source was replaced; restart processing for the new version')


def run(alias, args, log):
    verify(alias)
    print(alias, 'running', Path(args[1]).name, flush=True)
    with log.open('w') as output:
        subprocess.run([str(x) for x in args], env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    verify(alias)


for alias in ['crunch', 'pickup']:
    entry = status['clips'][alias]
    work = V2M / 'work/skills' / alias
    work.mkdir(parents=True, exist_ok=True)
    clip = alias+'_reference'
    entry['work_dir'] = str(work)
    try:
        entry['status'] = 'waiting_for_extraction'; save()
        while True:
            verify(alias)
            log = V2M / f'work/{alias}_pipeline.log'
            if log.exists():
                text = log.read_text(errors='replace')
                if f"Source fingerprint verified: {alias} {entry['sha256']}" in text:
                    break
                if 'Traceback (most recent call last)' in text:
                    raise RuntimeError(f'{alias}: extraction failed; inspect {log}')
            time.sleep(10)
        entry['status'] = 'preparing_natural_transitions'; save()
        preparation = [deploy_python, V2M/'prepare_skill.py', '--input', V2M/f'work/main/npz/{alias}.npz',
                       '--name', clip, '--work-dir', work, '--transitions', 'none']
        run(alias, preparation, work/'prepare_natural.log')
        evaluation = [deploy_python, V2M/'evaluate_skill.py', '--clip', clip, '--work-dir', work, '--track-group', 'all']
        entry['status'] = 'simulating_natural_transitions'; save()
        run(alias, evaluation, work/'simulation_natural.log')
        nominal = json.loads((work/'e11c_0220000_lat0.json').read_text())
        entry['synthetic_transitions_added'] = False
        start_end_failure = nominal.get('fall_phase') in ['interp', 'hold', 'standby_pre', 'standby_post']
        start_end_failure |= nominal.get('r1x_never', False)
        start_end_failure |= not nominal.get('smooth', False)
        start_end_failure |= not nominal.get('standby_ready', False)
        if not nominal['accepted'] and start_end_failure:
            for suffix in ['.json', '.npz']:
                p = work / ('e11c_0220000_lat0'+suffix)
                p.rename(work / ('natural_e11c_0220000_lat0'+suffix))
            (work/'preparation.json').rename(work/'natural_preparation.json')
            entry['status'] = 'testing_needed_standing_transitions'; save()
            preparation[-1] = 'standing'
            run(alias, preparation, work/'prepare_standing.log')
            run(alias, evaluation, work/'simulation_standing.log')
            nominal = json.loads((work/'e11c_0220000_lat0.json').read_text())
            entry['synthetic_transitions_added'] = True
        entry['nominal_pass'] = bool(nominal['accepted'])
        entry['nominal_upstream_whole_phase_pass'] = bool(nominal['pass'])
        if entry['nominal_pass']:
            entry['status'] = 'checking_command_delay'; save()
            run(alias, evaluation+['--latency-ms', '10'], work/'simulation_latency10.log')
            delayed = json.loads((work/'e11c_0220000_lat10.json').read_text())
            entry['delay_pass'] = bool(delayed['accepted'])
            entry['delay_upstream_whole_phase_pass'] = bool(delayed['pass'])
        entry['status'] = 'rendering_simulation'; save()
        run(alias, [render_python, V2M/'render_skill_sim.py', '--clip', clip, '--alias', alias, '--work-dir', work], work/'render_simulation.log')
        entry['preview'] = str(work/(alias+'_policy_vs_reference.webm'))
        entry['clip'] = clip
        entry['status'] = 'ready_for_review' if entry['nominal_pass'] and entry.get('delay_pass') else 'simulation_failed'
        save()
        print(alias, entry['status'], flush=True)
    except Exception as exc:
        entry.update(status='error', error=str(exc)); save()
        print(alias, 'ERROR', exc, flush=True)
print('Both clip jobs finished; bank registration remains a review step.', flush=True)
