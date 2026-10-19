# Reproducing CAD–stereo point-cloud construction and robotic grasping

This directory contains the frozen v6 fusion implementation, the input adapters
for five grasp generators, and preparation, execution and evidence-audit tools
for the current paper. The published result files are under
`RQ1/data/paper_final_20261019`, `RQ2/data/paper_final_20261019` and
`RQ3/data/paper_final_20261019`. New executions use a separate workspace.

## Experiment profiles and selected seeds

| Profile | Point-cloud inputs | Unique grasp candidate tasks | Physical outcomes |
|---|---:|---:|---:|
| `rq1` | 5,568 | 0 | 0 |
| `rq2_perception` | 5,568 | 5,904 | 11,136 |
| `rq2_cross_grasp` | 1,392 | 8,688 | 13,920 |
| `rq3` | 2,784 | 3,120 | 5,568 |

All profiles cover 696 scenes from 30 T-LESS and 28 ITODD objects. Each object has
12 predefined scenes. There are 29 original batches; their scene identities and
batch mapping are retained. RQ1 evaluates the seven stereo methods and fused CAD
geometry without generating grasp candidates or running grasp physics. Rendering
its new stereo images still requires Isaac Sim.

`rq2_perception` couples StereoBM, SGM, PSMNet, GwcNet, RAFT-Stereo, IGEV-Stereo,
FoundationStereo and CAD fusion to GraspGen. `rq2_cross_grasp` crosses
FoundationStereo and CAD fusion with GraspGen, 6-DoF GraspNet,
Contact-GraspNet, GPD and GraspNet-Baseline. `rq3` compares FoundationStereo,
registered CAD, direct concatenation and v6 fusion with GraspGen.

The registered-CAD and concatenation ablations use the same estimated CAD pose
and registration acceptance gate as v6. Registered-CAD still uses stereo for
alignment and observed obstacles. A rejected gate remains a failed scheduled
task for these two ablations. Original-size CAD is retained; only millimetres
are converted to metres. Registration estimates tabletop translation and yaw
under a known support orientation.

`RQ2/data/paper_final_20261019/selection.json` records the ten conditions for
which complete rounds were compared. `configs/execution_seeds.json` binds that
file's SHA-256 and explicitly registers all twenty dataset/platform/grasp CAD
conditions, including the ten unchanged original seed-zero conditions. It was
checked against every CAD row in the published `cad_comparisons.jsonl`.
Missing conditions or conflicting seeds cause preparation to fail. All
non-CAD comparison arms use their original seed zero.

These are the published whole-round choices, including ITODD/MuJoCo/GraspGen
seed 1 and selected additional-backend seed 3 conditions. They are descriptive
best-observed comparisons after targeted CAD-only repeats. This runner executes
the chosen rounds; it does not search for a better seed or change the success
criteria. The same scene/perception/grasp/seed shares one candidate set and path
between simulators. Different selected seeds create distinct candidate tasks
and are dispatched only to their registered simulator.

The profiles overlap. Their outcome totals must not be added as independent
paper observations. Each profile is prepared independently for reproduction;
the published data separately record the paper's reuse.

## Numerical code and dependencies

`runtime/` contains v6 registration/fusion, point-cloud preprocessing,
candidate generation, Panda planning and both physics workers. The older
version identifiers in `geometry_v6.py` and `refined_v6.py` identify live
dependencies of `adaptive_fusion_v6.py`. `RQ1/` holds the seven-method stereo
driver, `RQ2/` holds task design and CAD input adaptation, and `RQ3/` holds the
registered-CAD/direct-concatenation implementations. Shared evidence auditing
and aggregate metrics are at this directory's root.

`configs/code_provenance.json` records the final file hashes and preserved
numerical functions. Portability changes concern paths, task identities,
platform/seed dispatch, input metadata and evidence bookkeeping. The CAD grasp
adapter preserves every array in the fused cloud exactly. It retains the
observed points' camera IDs, assigns appended fusion points camera ID -1, and
copies the observed camera origins and reference transform. It does not invent
CAD colors or normals or modify model weights.

Run `python3 dependencies.py` to inspect pinned source repositories without
downloading them. `python3 dependencies.py --fetch-sources` explicitly fetches
the recorded revisions from `configs/upstream_sources.json`. Acquire model
weights separately under their licenses, then use
`python3 dependencies.py --check-weights` with
`configs/model_weights.json`. Source download alone does not install models,
compiled operators or simulators.

Experiments require Linux with a compatible NVIDIA GPU. The recorded hardware
was RTX 4090 with 24 GiB; the robot is Franka Emika Panda, with Isaac Sim
5.1.0.0 / Isaac Lab 2.3.2.post1 and MuJoCo 3.3.7. Observed package versions are
in `configs/environments.json`; this record is not a full resolver lock.
Maintain separate `.venv-infer`, `.venv-mujoco` and `.venv-sim` environments
below `GRASPGEN_RUNTIME`. Alternative grasp methods use the environment
locations in `dependencies/grasp_modules/shared/infer.py`.

Install the pinned authors' software and build the supplied TensorFlow, GPD
and GraspNet compatibility operators before execution. The build and validation
helpers are under `dependencies/grasp_modules`. Set `CUDA_HOME` when building
the TensorFlow operators. GPD's compatibility patch retains its upstream
license. GraspNet-Baseline's checkpoint mirror provenance is recorded alongside
its model metadata; identity with the inaccessible official download was not
independently established.

## Configure paths and restore CAD inputs

Run these commands from this `Source_Codes` directory:

```sh
export CAD_RELEASE_ROOT="$PWD"
export CAD_WORKSPACE="/absolute/path/to/new-workspace"
export GRASP_MODULE_ROOT="$PWD/dependencies/grasp_modules"
export STEREO_MODULE_ROOT="$PWD/dependencies/stereo_modules"
export GRASPGEN_RUNTIME="$GRASP_MODULE_ROOT/baseline_graspgen"
export CAD_GPU_LOCK="$CAD_WORKSPACE/gpu.lock"
```

Acquire and unpack the CAD archives identified by
`configs/dataset_sources.json`, with directories `tless/models_cad` and
`itodd/models` under the chosen original-data directory. Dataset terms are in
`datasets/ATTRIBUTION.md` and `datasets/third_party_data.json`.
T-LESS uses CC BY 4.0; ITODD uses CC BY-NC-SA 4.0. Original meshes, derived CAD
arrays, upstream repositories, weights and full physics traces are not bundled
in the public source package.

```sh
python3 prepare_assets.py --original /absolute/path/to/original \
  --workspace "$CAD_WORKSPACE"
```

Asset reconstruction uses NumPy 1.26.4 and trimesh 4.5.3 and requires exact
original and derived hashes from `datasets/canonical_catalog.json`.
Serialization changes must be resolved rather than replacing frozen hashes.
The script retains original-size geometry and reproduces CAD-only support
transforms. Current local inputs may remain in `datasets/original/` and
`datasets/canonical_assets/`; both are excluded from Git. Reconstruction uses
`datasets/assets/` for intermediate metric meshes and
`datasets/canonical_assets/` for final inputs.

The released catalog is byte-identical to the current frozen local catalog.
Its historical `simulator_preflight` fields describe asset provenance, not the
current experiment completion state. Preparation validates all required mesh
and sample hashes rather than interpreting these old status fields.

## Prepare, run and audit

```sh
python3 prepare_experiment.py --workspace "$CAD_WORKSPACE" \
  --menagerie "$PWD/dependencies/assets/menagerie" --batch 0 \
  --profile rq2_cross_grasp
python3 run_experiment.py \
  --run "$CAD_WORKSPACE/rq2_cross_grasp_paper_final/batches/batch_000" --list
python3 run_experiment.py \
  --run "$CAD_WORKSPACE/rq2_cross_grasp_paper_final/batches/batch_000"
```

Choose another profile with `--profile`, or one simulator with
`--platform franka_mujoco` or `--platform franka_isaac` during preparation.
Use the same platform selection when aggregating. Repeat batch preparation and
execution for indices 0 through 28. Existing preparations and stage receipts
are preserved and rejected. `--stage NAME` executes only the named stage after
its prerequisites have been completed; interrupted attempts require diagnosis
and an explicit recovery decision.

The complete grasp pipeline calibrates each selected simulator, renders and
freezes the stereo inputs, computes point clouds, generates candidates, plans
paths, executes physics and audits evidence. Inference uses a network-isolated
Linux namespace via `sudo -n unshare` and `setpriv`; configure that privilege
policy locally and install all sources and weights beforehand. The runner
does not grant itself those privileges. Evaluation geometry is excluded from
the offline inference path.

GPU ownership belongs to the inference, rendering or Isaac worker. The runner
does not hold a GPU lock while waiting for those children. The physics
controller probes GPU availability, closes that probe lock, and then launches
the Isaac worker. Its per-platform controller lock is a different file.

After all 29 batches pass their audits:

```sh
python3 summarize_experiment.py --workspace "$CAD_WORKSPACE" \
  --profile rq2_cross_grasp
```

The aggregate is written to the workspace profile's `results/summary.json`.
It rejects missing/duplicate scene-method-platform outcomes and wrong seeds.
Object success rate is the number of objects with at least one successful
scene divided by all dataset objects. Attempt success rate divides successful
scenes by all registered scenes; perception/candidate failures and no-path
outcomes remain in that denominator. Datasets and simulators are reported
separately. Point-cloud metrics are evaluated once per scene/perception.

## Packaging verification scope

CPU validation checked all profile counts and twenty CAD seed routes,
single-platform restriction, missing/wrong seed rejection, preparation and CLI
listing, the hashes and readability of all 58 canonical meshes and sample
arrays, CAD metadata adaptation and its failure check, and GPU lock ownership
using real CPU subprocesses. Numerical fusion/planning/physics files and the
geometry/source-ablation functions were compared against the previous
verified release.

No new GPU inference, scene rendering or physical experiment was performed for
this packaging change. CPU fixtures are verification inputs, not experimental
observations. New end-to-end runs require the external environments and assets
above and may differ numerically across software and hardware installations.
