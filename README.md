# CAD Geometry and Stereo Depth Fusion for Robotic Grasping

This repository contains the implementation and experiment data for a study of
**point cloud construction using known CAD geometry and stereo depth**, evaluated
through robotic grasping. The method combines FoundationStereo observations with
sampled CAD surfaces, aligns the two point clouds, and uses confidence constraints
to correct and complete object geometry. Grasp generation, motion planning, and
physics simulation evaluate the resulting point clouds.

The current release is `paper_final_20261019`. It provides the frozen v6 fusion
method, experiment configurations, per-scene outcomes, provenance records, and
scripts that recompute the reported results.

## Verify the reported results

```sh
git clone https://github.com/warmjademe/robotic_grasping_CAD_IntegratingWith_Foundationstereo.git
cd robotic_grasping_CAD_IntegratingWith_Foundationstereo
python3 Source_Codes/scripts/verify_paper_results.py
```

The verifier requires only the Python standard library. It checks released file
hashes, scene denominators, selected rounds, shared outcomes, and numerical
results. It does not download models or start a simulation.

For new experiments, follow the [execution guide](Source_Codes/REPRODUCING.md).
The [release guide](REPRODUCE.md) explains the result selection and evidence scope.

## Method

1. **Construct and align two point clouds.** FoundationStereo estimates depth
   from calibrated stereo images. Surface sampling produces a point cloud from
   the known object CAD mesh. Registration estimates tabletop translation and yaw
   under a known support orientation.
2. **Fuse geometry using confidence constraints.** Observation reliability,
   registration consistency, and visibility determine which surfaces to retain,
   correct, or complete. Rejected fusion conditions fall back to the observed
   point cloud.
3. **Evaluate downstream grasping.** Frozen grasp generators receive the resulting
   geometry and use a common planning, control, and success-evaluation pipeline.

Registration uses a support-orientation prior; grasp candidates are full
six-degree-of-freedom poses. CAD models retain their original dimensions, with
millimetres converted to metres.

## Study design

- **Datasets:** all 30 T-LESS and 28 MVTec ITODD object types.
- **Scenes:** 12 scenes per object, combining four placements with three
  conditions: isolated object, neighbouring object, and bin wall. The 696 scenes
  use simulated stereo images generated from public CAD assets.
- **Robot and simulators:** Franka Emika Panda, Isaac Sim 5.1.0.0 with Isaac Lab
  2.3.2.post1, and MuJoCo 3.3.7. The recorded GPU was an NVIDIA RTX 4090 (24 GiB).
- **RQ1 — Point cloud quality:** compare fusion with seven stereo methods:
  StereoBM, SGM, PSMNet, GwcNet, RAFT-Stereo, IGEV-Stereo, and FoundationStereo.
- **RQ2 — Grasping performance:** compare those perception methods with GraspGen,
  then compare FoundationStereo and CAD-fused inputs within GraspGen,
  6-DOF GraspNet, Contact-GraspNet, GPD, and GraspNet-Baseline.
- **RQ3 — Ablation:** hold GraspGen fixed and compare stereo only, registered CAD
  target geometry, direct concatenation, and full fusion. The registered-CAD arm
  still uses stereo for alignment and observed obstacles.

The original eleven baseline configurations comprise seven stereo methods with
GraspGen and four alternative grasp generators with FoundationStereo.

## Grasp success rates

**Object-level grasp success rate** is the proportion of object types with at
least one fully successful grasp among their twelve predefined scenes. This is
the primary metric. **Scene-level grasp success rate** is the proportion of
successful scenes among all predefined scenes. Failed grasps, rejected inputs,
and no-path cases remain in the denominator.

A successful execution satisfies the recorded endpoint tolerance (2 mm and
0.5 degrees), lifts the object bottom by at least 100 mm for two seconds, and
satisfies the prohibited-contact force limit of 0.5 N. Evaluation records the
complete execution, including the return phase.

The selected full-fusion results with **GraspGen** are:

| Dataset | Simulator | Object-level success | Scene-level success |
|---|---|---:|---:|
| T-LESS | Isaac Sim | 96.7% (29/30) | 80.6% (290/360) |
| T-LESS | MuJoCo | 56.7% (17/30) | 21.1% (76/360) |
| ITODD | Isaac Sim | 82.1% (23/28) | 64.0% (215/336) |
| ITODD | MuJoCo | 53.6% (15/28) | 14.6% (49/336) |

Relative to FoundationStereo, mean surface error decreased from 2.508 to
1.957 mm on T-LESS (22.0%) and from 2.479 to 1.376 mm on ITODD (44.5%). Across
twenty CAD/no-CAD comparisons with five grasp generators, object-level success
increased in sixteen conditions, tied in two, and decreased in two. Scene-level
success increased in fourteen, tied in one, and decreased in five.

Full fusion achieved the highest object-level success in three of the four
ablation settings. On T-LESS/MuJoCo, direct concatenation covered 18/30 objects,
compared with 17/30 for fusion; registered CAD achieved 91/360 successful scenes,
compared with 76/360 for fusion.

For each CAD dataset/simulator/grasp-generator condition, the release retains one
complete eligible round, selected by successful objects, then successful scenes,
then the lower seed. CAD conditions received targeted additional runs; comparator
arms retain single runs. These are descriptive best-observed comparisons with
different selection budgets. Every selected round includes all its scenes.
See the [selection record](Source_Codes/RQ2/data/paper_final_20261019/selection.json)
and [data guide](Source_Codes/data/PAPER_RESULTS_README.md).

## Repository layout

| Path | Contents |
|---|---|
| [Source_Codes/RQ1](Source_Codes/RQ1) | Stereo comparison driver and point cloud quality data |
| [Source_Codes/RQ2](Source_Codes/RQ2) | Grasping comparisons, CAD input adapters, and selected outcomes |
| [Source_Codes/RQ3](Source_Codes/RQ3) | Registered-CAD and concatenation ablations and their outcomes |
| [Source_Codes/runtime](Source_Codes/runtime) | Fusion, preprocessing, grasping, planning, and simulation workers |
| [Source_Codes/configs](Source_Codes/configs) | Environment records, upstream sources, assets, and selected seeds |
| [Source_Codes/scripts](Source_Codes/scripts) | Offline verification of released paper results |
| [Source_Codes/data](Source_Codes/data) | Result index, hashes, provenance, and shared-observation records |
| [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) | File sizes and SHA-256 hashes for the public release |

The release contains 25,056 distinct grasping outcome records across overlapping
research-question datasets. Shared observations are identified in the
[result index](Source_Codes/data/paper_results_index.json) and do not represent
additional physical executions.

## External assets and terms

Original CAD meshes, generated images and point cloud arrays, model weights,
and full physical trajectories are stored separately. The execution guide lists
the required assets, pinned upstream sources, and preparation commands. The
published JSON/JSONL results can be verified without these large assets.

T-LESS is distributed under CC BY 4.0; MVTec ITODD uses CC BY-NC-SA 4.0.
See [dataset attribution](Source_Codes/datasets/ATTRIBUTION.md). Upstream code,
pretrained models, and datasets retain their respective terms; this documentation
does not assign a new project-wide license.
