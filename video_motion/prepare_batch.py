"""Retarget and simulate a fingerprinted video batch with the existing G1 tools.

Expects upright 30 fps videos and the corresponding cloud pose caches. Exports
clips under the video filename stems. Does not start a controller or edit the bank.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
V2M = ROOT / 'video_motion'
REPO = ROOT / 'SuperMimic-Repro-Sim2Real'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--work-dir', type=Path, required=True)
    ap.add_argument('--names', nargs='+')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--wait-for-cache', action='store_true')
    ap.add_argument('--model', default='e11c_0220000')
    a = ap.parse_args()
    work = a.work_dir.resolve()
    clips = json.loads(a.manifest.read_text())['clips']
    names = a.names or list(clips)
    env = os.environ.copy()
    env.update(G1_DEPLOY_HOME=str(ROOT / 'g1_deploy'),
               G1_UNITREE_XML=str(ROOT / 'g1_deploy/unitree_mujoco/unitree_robots/g1/scene_29dof.xml'),
               GMR_ROOT=str(V2M / 'GMR'), SMPLX_BODY_MODELS=str(V2M / 'models'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    env['LD_LIBRARY_PATH'] = str(ROOT / 'g1_deploy/cyclonedds/lib') + ':' + str(ROOT / 'g1_deploy/venv/lib')
    env['PATH'] = '/home/yiannis/natalia/mamba/envs/vlaclient/bin:' + env['PATH']
    env.pop('PYTHONPATH', None)
    deploy = ROOT / 'g1_deploy/venv/bin/python'
    gmr = V2M / 'gmr_env/bin/python'
    xml = ROOT / 'g1_deploy/mjlab-assets/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'

    def process(name):
        entry = dict(clips[name])
        target = work / 'skills' / name
        target.mkdir(parents=True, exist_ok=True)

        def save(stage, **extra):
            entry.update(status=stage, **extra)
            tmp = target / 'status.json.tmp'
            tmp.write_text(json.dumps(entry, indent=2) + '\n')
            tmp.replace(target / 'status.json')
            print(f'{name}: {stage}', flush=True)

        def verify():
            for path, key in [(ROOT / entry['source'], 'source_sha256'),
                              (work / 'video30' / (name + '.mp4'), 'normalized_sha256')]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry[key]:
                    raise ValueError(f'{path}: fingerprint mismatch')
            marker = work / 'gvhmr' / name / 'input.sha256'
            if marker.read_text().strip() != entry['normalized_sha256']:
                raise ValueError(f'{name}: pose cache fingerprint mismatch')

        def run(args, log):
            verify()
            with (target / log).open('w') as output:
                subprocess.run([str(x) for x in args], env=env, stdout=output,
                               stderr=subprocess.STDOUT, check=True)
            verify()

        try:
            if a.wait_for_cache:
                save('waiting_for_pose_cache')
                deadline = time.monotonic() + 3600
                while not (work / 'gvhmr' / name / 'hmr4d_results.pt').exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError('Pose cache did not arrive within one hour')
                    time.sleep(5)
            verify()
            save('retargeting')
            csv = work / 'csv' / (name + '.csv')
            run([gmr, REPO / 'tools/video2motion/gvhmr_to_supermimic.py',
                 '--gvhmr_pred', work / 'gvhmr' / name / 'hmr4d_results.pt',
                 '--name', name, '--floor_mode', 'drift', '--out_dir', csv.parent,
                 '--robot_xml', xml, '--init_from_human', '--restart_tol', '0.1'], 'retarget.log')
            save('converting')
            npz = work / 'npz' / (name + '.npz')
            run([deploy, '-c',
                 'import sys; sys.path.insert(0, sys.argv[4]); '
                 'from supermimic.data.motion_convert import load_g1_model, convert_csv_to_npz; '
                 'print(convert_csv_to_npz(sys.argv[1],sys.argv[2],30,50,model=load_g1_model(sys.argv[3])))',
                 csv, npz, xml, REPO / 'src'], 'convert.log')
            prepare = [deploy, V2M / 'prepare_skill.py', '--input', npz, '--name', name,
                       '--work-dir', target, '--model', a.model, '--transitions', 'none']
            evaluate = [deploy, V2M / 'evaluate_skill.py', '--clip', name,
                        '--work-dir', target, '--model', a.model, '--track-group', 'all']
            save('preparing_natural_transitions')
            run(prepare, 'prepare_natural.log')
            save('simulating_natural_transitions')
            run(evaluate, 'simulation_natural.log')
            nominal = json.loads((target / f'{a.model}_lat0.json').read_text())
            endpoint_failure = nominal.get('fall_phase') in ('interp', 'hold', 'standby_pre', 'standby_post')
            endpoint_failure |= nominal.get('r1x_never', False) or not nominal.get('smooth', False)
            endpoint_failure |= not nominal.get('standby_ready', False)
            if not nominal['accepted'] and endpoint_failure:
                for filename in ('preparation.json', f'{a.model}_lat0.json', f'{a.model}_lat0.npz'):
                    (target / filename).replace(target / ('natural_' + filename))
                save('adding_standing_transitions')
                prepare[-1] = 'standing'
                run(prepare, 'prepare_standing.log')
                run(evaluate, 'simulation_standing.log')
                nominal = json.loads((target / f'{a.model}_lat0.json').read_text())
            save('simulating_10ms_delay')
            run(evaluate + ['--latency-ms', '10'], 'simulation_latency10.log')
            delayed = json.loads((target / f'{a.model}_lat10.json').read_text())
            preparation = json.loads((target / 'preparation.json').read_text())
            save('rendering', nominal_accepted=nominal['accepted'], delay_accepted=delayed['accepted'],
                 transitions=preparation['transitions_used'], frames=preparation['frames'],
                 duration_s=preparation['duration_s'], clip=name, model=a.model)
            run([gmr, V2M / 'render_skill_sim.py', '--clip', name, '--alias', name,
                 '--work-dir', target, '--model', a.model], 'render.log')
            save('validated' if nominal['accepted'] and delayed['accepted'] else 'simulation_failed',
                 preview=str((target / (name + '_policy_vs_reference.webm')).relative_to(ROOT)))
        except Exception as exc:
            save('error', error=str(exc))
        return name, entry

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = dict(pool.map(process, names))
    summary = {}
    for name in clips:
        path = work / 'skills' / name / 'status.json'
        if path.exists():
            entry = json.loads(path.read_text())
            if all(entry.get(k) == clips[name][k] for k in ('source_sha256', 'normalized_sha256')):
                summary[name] = entry
    (work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if any(e['status'] == 'error' for e in results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
