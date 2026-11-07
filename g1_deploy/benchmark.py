"""Offline CPU inference benchmark; never connects to DDS or the robot."""
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/yiannis/policies/SuperMimic-Repro-Sim2Real/src')
from supermimic.deploy.core import OnnxPolicy
repo = Path('/home/yiannis/policies/SuperMimic-Repro-Sim2Real')
results = {}
for name in ('e13i_0291000', 'e24_0355000'):
    model = repo / 'deploy_artifacts' / name / 'policy.onnx'
    if not model.exists():
        continue
    policy = OnnxPolicy(model)
    obs = np.zeros(930, np.float32)
    goal = np.zeros(680, np.float32)
    times = []
    for i in range(1100):
        start = time.perf_counter()
        action = policy(obs, goal)
        elapsed = (time.perf_counter() - start) * 1000
        assert action.shape == (29,) and np.isfinite(action).all()
        if i >= 100:
            times.append(elapsed)
    results[name] = {'samples': len(times), 'p50_ms': float(np.percentile(times, 50)),
                     'p99_ms': float(np.percentile(times, 99)), 'max_ms': max(times),
                     'policy_period_ms': 20, 'providers': policy.sess.get_providers()}
print(json.dumps(results, indent=2))
Path('/home/yiannis/policies/g1_deploy/logs/inference-benchmark.json').write_text(json.dumps(results, indent=2)+'\n')
