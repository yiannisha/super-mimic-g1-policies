# SuperMimic G1 policies

SuperMimic policy deployment, live motion switching, Dex1 gripper control, and video-to-motion tooling for a Unitree G1. The deployment setup records a Jetson AGX Orin host; GPU pose extraction also has a separate cloud workflow.

## Repository layout

| Path | Contents |
| --- | --- |
| `SuperMimic-Repro-Sim2Real/` | SuperMimic source with local deployment, live switching, and gripper changes; upstream licenses and documentation are retained. |
| `g1_deploy/` | Host setup notes, launch scripts, pinned runtime requirements, gripper calibration, and motion preparation tools. |
| `video_motion/` | Local and cloud video extraction, retargeting, skill preparation, evaluation, and rendering scripts. GVHMR includes local CPU/device-selection changes. |
| `setup/` | Original host installation status, resume script, and package/environment inventories. |
| `Isaac-GR00T/` | Pinned upstream checkout for the separate GR00T setup. |
| `upstream-sources.json` | Upstream URLs and exact source revisions for imported source and dependencies. |

## Clone and dependencies

```bash
git clone --recurse-submodules git@github.com:yiannisha/super-mimic-g1-policies.git
cd super-mimic-g1-policies
# For an existing clone:
git submodule update --init --recursive
```

Unmodified upstream dependencies are submodules. SuperMimic and GVHMR are tracked source snapshots so local changes are available directly in this repository. Their original revisions are recorded in `upstream-sources.json`; their licenses continue to apply. GVHMR's DPVO dependency is registered in the root `.gitmodules`.

## Deployment and video workflows

Read [`g1_deploy/SETUP.txt`](g1_deploy/SETUP.txt), [`g1_deploy/motion_suite/USAGE.txt`](g1_deploy/motion_suite/USAGE.txt), and [`g1_deploy/motion_suite/LIVE_RUNNER.txt`](g1_deploy/motion_suite/LIVE_RUNNER.txt) for the existing deployment setup and motion controls. Follow [`SuperMimic-Repro-Sim2Real/docs/DEPLOY_G1.md`](SuperMimic-Repro-Sim2Real/docs/DEPLOY_G1.md) for hardware commissioning and operator procedures. Calibration is specific to the original robot.

For video processing, see `video_motion/run-main.sh` and [`video_motion/cloud/USAGE.txt`](video_motion/cloud/USAGE.txt). Local environment inventories are preserved in `video_motion/requirements-gmr-lock.txt` and `video_motion/requirements-gvhmr-lock.txt`. The three root MP4 files are source inputs. Motion reference clips, gripper sidecars, and deployment configuration JSON files are tracked under `SuperMimic-Repro-Sim2Real/deploy_artifacts/`.

Additional source recordings live in `raw_videos/`. Their motion aliases use the filename stems. `video_motion/prepare_batch.py` prepares these batches from cloud pose caches using the existing retargeting and simulation tools; source fingerprints and per-motion results are recorded in `video_motion/raw_video_skills.json`. Generated simulation traces and previews live under `video_motion/work/raw_batch/skills/`.

Virtual environments, installed CycloneDDS libraries, compiled bridge binaries, downloaded archives and model weights, body models, caches, and generated logs/previews remain local. Obtain SuperMimic policy bundles from the [upstream releases](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/releases), verify their supplied checksums, and restore them under `SuperMimic-Repro-Sim2Real/deploy_artifacts/`. GVHMR checkpoint retrieval is described in `video_motion/cloud/USAGE.txt` and pinned by `video_motion/cloud/checkpoints.json`. Obtain SMPL/SMPL-X body models separately under their applicable access terms and place them in `video_motion/models/`.

The launch scripts and setup notes retain paths from `/home/yiannis/policies`, including references to environments and the Unitree Python SDK outside this repository. Adjust those paths and install the recorded dependencies on a new host. Build the C++ bridge using `SuperMimic-Repro-Sim2Real/scripts/deploy/bridge/build.sh` after restoring the SDK dependencies. A fresh clone alone is not a complete runtime installation.

Setup and validation results describe the original host at the recorded time. Consult the deployment notes for remaining hardware acceptance requirements.
