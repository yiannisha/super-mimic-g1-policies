"""Reconstruct the released 46-second dance recipe without the training stack."""
import importlib.util
import json
import numpy as np
from common import HOME, REPO, mc, model, export


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = REPO / 'deploy_artifacts/e13i_0291000/deploy_config.json'
raw = REPO / 'data/npz/custom_dance'
combo = REPO / 'data/npz/custom_combo'
raw.mkdir(parents=True, exist_ok=True)
combo.mkdir(parents=True, exist_ok=True)
m = model()
for name in ('shanshenbu', 'langzi_tiqiu', 'gouxiong_duosuomao', 'yaotou_huangnao'):
    result = mc.convert_csv_to_npz(REPO / 'motions/custom_dance' / (name+'.csv'), raw / (name+'.npz'), 30, model=m)
    print(json.dumps(result), flush=True)
stance_csv = REPO / 'data/raw/custom_dance/stand_idle.csv'
stance_csv.parent.mkdir(parents=True, exist_ok=True)
original = REPO / 'motions/custom_dance/shanshenbu_combo4.csv'
stance_csv.write_text(''.join(original.read_text().splitlines(keepends=True)[:30]))
print(mc.convert_csv_to_npz(stance_csv, raw / 'stand_idle.npz', 30, model=m), flush=True)
cap = module('upstream_cap', REPO / 'scripts/deploy/cap_joint_velocity.py')
cap.main(src=raw/'gouxiong_duosuomao.npz', dst=raw/'gouxiong_duosuomao_vc.npz', deploy_config=cfg)
make = module('upstream_combo', REPO / 'scripts/deploy/make_combo_clip.py')
make.load_g1_model = model
make.main(npz_dir=raw, out_dir=combo,
          clips=('stand_idle','shanshenbu','langzi_tiqiu','gouxiong_duosuomao_vc','yaotou_huangnao','stand_idle'),
          name='combo_c4_full_vc_b15f20', blend_s=1.5, last_blend_s=2.0, overwrite=True)
result = export(combo/'combo_c4_full_vc_b15f20.npz', REPO/'deploy_artifacts/clips/combo_c4_full_vc_b15f20.npz', cfg)
print(json.dumps(result), flush=True)
(HOME/'logs/dance46-preparation.json').write_text(json.dumps(result, indent=2)+'\n')
