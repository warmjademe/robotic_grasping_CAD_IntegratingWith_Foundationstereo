# CAD geometric priors and stereo depth for point-cloud construction

This repository contains the selected final implementation and recorded experiment
data for CAD and stereo point-cloud fusion, evaluated in robotic grasping software.
The method aligns sampled CAD geometry with FoundationStereo observations and
fuses them under observation, registration and visibility constraints. Frozen
GraspGen, planning and physical execution provide the downstream application.

The release uses frozen fusion v6 and the final single-seed experiment. It excludes
superseded development variants and unfinished historical controls. Necessary
runtime dependencies are retained even when their filenames contain an older
version number. The original research archives are preserved separately.

## Read and verify the results

```sh
python3 results/verify_release.py
```

This standard-library check does not access the network or start a simulation.
It checks every registered outcome, recomputes success rates and paired tests,
checks shared results, and verifies the released data hashes.

- [Data schema and metric definitions](results/SCHEMA.md)
- [Main comparison](results/comparison_seed0_v1/summary.json)
- [Source ablation](results/rq3_source_ablation_seed0_v1/summary.json)
- [Audit scope and current review status](results/review_status.json)
- [Input dataset versions and hashes](data/dataset_sources.json)

The main comparison includes 16,704 terminal outcomes: 58 objects, 12 scenes per
object, 12 configurations and two simulators, with one run per configuration and
scene. The 11 baselines comprise seven perception methods coupled to GraspGen
and four other grasp generators coupled to FoundationStereo. FoundationStereo
with GraspGen is the no-CAD reference among those seven perception baselines.

The source ablation contains 5,568 outcomes, including 2,784 reused main-comparison
outcomes. There are 19,488 unique outcomes across the studies. Failures, rejected
inputs and no-path results remain in their denominators. The original 29 ablation
batch audits and a fresh source-hash check passed; the final manuscript analysis
of the ablation remains pending.

The primary grasp success rate is the proportion of objects with at least one
fully successful grasp in their 12 predefined scenes. Scene success rate is
reported separately. T-LESS and ITODD, and Isaac Sim and MuJoCo, are not pooled.
Point-cloud quality and physical-task success are distinct measurements.

## Scope of the implementation

See [the reproduction instructions](docs/REPRODUCING.md) for external dependencies,
asset reconstruction, preparation and execution commands. To inspect pinned
dependencies without downloading anything, run `python3 code/dependencies.py`.

`code/` contains the fusion, inference adapters, planning/execution and evaluation
implementation. `configs/` fixes experiment settings and external dependency
identities. `dependencies/` contains required local adapters and build scripts,
not bundled upstream model repositories or weights. `data/` holds all 696 scene
descriptions and source metadata; `results/` holds complete terminal results,
geometry evaluations, aggregates and raw-evidence hash inventories.

Experiments use Franka Emika Panda in Isaac Sim 5.1 and MuJoCo 3.3.7. CAD scale is
preserved, converting millimetres to metres. Registration estimates tabletop
translation and yaw under a known support orientation. This is not an arbitrary
six-degree-of-freedom object localization method. Grasp candidates are six-degree-
of-freedom poses.

The released statistics can be checked without GPU software. Re-executing
inference and physics requires the separately installed simulator environments,
pinned upstream code, licensed weights and dataset downloads. Publication checks
do not constitute a new end-to-end GPU rerun or promise bitwise identical physics.

## External assets and publication boundaries

Original CAD meshes, synthetic images, point-cloud arrays, model weights and
large physics traces are not included. The primary and ablation working
directories total approximately 45 GB. Source identifiers and SHA-256 inventories
trace the retained evidence; no public download endpoint for those omitted raw
traces is provided by this release. Pinned CAD downloads and code support asset
regeneration under the applicable upstream terms.

T-LESS is distributed under CC BY 4.0; MVTec ITODD uses CC BY-NC-SA 4.0. See
[dataset attribution and license links](data/third_party_data.json) and
[pinned upstream source licenses](configs/upstream_sources.json). Code, model and
asset licenses can differ. No new project-wide license is selected by this
publication; third-party terms remain applicable.

The manuscript, original project README, handoff notes, goals, web dashboard,
operations records, credentials and superseded experimental snapshots are outside
this repository. This file is a new implementation and data guide, not a copy of
the excluded project README.
