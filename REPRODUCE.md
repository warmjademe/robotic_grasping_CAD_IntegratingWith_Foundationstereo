# CAD geometry and stereo depth for robotic grasping

This release contains the frozen v6 point cloud fusion implementation and the
final data used in the manuscript updated on 19 October 2026. The directory
`Source_Codes` is organized around the three research questions. Previous
development variants and superseded result tables have been removed from the
current release.

## Verify the manuscript results

```sh
python3 Source_Codes/scripts/verify_paper_results.py
```

This command uses only the Python standard library. It checks file hashes,
complete scene denominators, selected seeds, shared outcomes and the reported
metrics. It neither starts a simulator nor downloads models.

| Research question | Released data | Scope |
|---|---|---|
| RQ1: point cloud quality | `Source_Codes/RQ1/data/paper_final_20261019/` | 5,568 geometry records for eight perception methods |
| RQ2: grasping performance | `Source_Codes/RQ2/data/paper_final_20261019/` | Seven perception baselines with GraspGen; 13,920 outcomes for ten perception/grasping combinations |
| RQ3: source and fusion ablation | `Source_Codes/RQ3/data/paper_final_20261019/` | 5,568 outcomes for four ablation configurations and 2,784 geometry records |

The [result index](Source_Codes/data/paper_results_index.json) records file hashes,
source identities and overlapping observations. There are 25,056 distinct
released grasping trials across the research questions. Reused observations are
identified and are not additional physical executions.

The primary metric is **object-level grasp success rate**: an object type counts
as successful if at least one of its twelve predefined scenes succeeds. The
secondary metric is **scene-level grasp success rate** over all predefined
scenes. Each selected complete round retains failed grasps, rejected inputs and
no-path outcomes. T-LESS and ITODD and the two simulators are reported separately.

## Final result selection

The supplementary evaluation completed all 3,456 registered platform outcomes
with fixed new seed 3, including 420 compatible reused outcomes. The original
three-round continuation was shortened to one round. Previously completed
historical rounds remain eligible under the recorded selection rule.

For each dataset/platform/grasping-method condition, selection uses one complete
round: first the number of successful objects, then the number of successful
scenes, then the lower seed. Objects, scenes and metrics are not mixed between
rounds. Six newly evaluated conditions exceeded their prior complete-round
results. MuJoCo/GraspGen retains the previous T-LESS seed-0 result (17/30 objects,
76/360 scenes) and ITODD seed-1 result (15/28 objects, 49/336 scenes).

The twenty selected CAD/no-CAD comparisons have sixteen increases, two ties and
two decreases in object success. The additional evaluation targeted prior ties
or decreases and retained single-run comparators, so these are descriptive
comparisons with different selection budgets. The
[selection record](Source_Codes/RQ2/data/paper_final_20261019/selection.json)
specifies the selection rule and source hashes. Unselected outcome rows are
kept in the research archive and are not bundled as current manuscript data.

## Execution and external assets

The source package includes the point cloud fusion, grasp-input adapters,
planning and simulation code, fixed configuration and scene definitions.
Models use frozen pretrained weights. The robot is a Franka Emika Panda with
its parallel-jaw gripper; the recorded experiments use Isaac Sim 5.1.0 and
MuJoCo 3.3.7. CAD dimensions are preserved with millimetres converted to metres.
The alignment estimates tabletop translation and yaw with a known support
orientation; grasp candidates are six-degree-of-freedom poses.

The [execution guide](Source_Codes/REPRODUCING.md) lists environment variables,
pinned upstream dependencies, asset preparation and execution commands. CPU checks
cover preparation, routing and result verification. The release preparation
does not constitute a new GPU experiment; the published values are the audited
recorded results.

Original CAD meshes, model weights, generated images and point cloud arrays,
and full physical trajectories are stored separately. Source identities,
download metadata and evidence hashes are included; this repository does not
provide a public endpoint for the omitted raw trajectories. Local CAD inputs
can remain under `Source_Codes/datasets/original` and
`Source_Codes/datasets/canonical_assets`; these large asset directories are
excluded from Git.

T-LESS uses CC BY 4.0 and MVTec ITODD uses CC BY-NC-SA 4.0. Dataset attribution
and upstream source records are included in the source package. Code, model
and dataset terms can differ. No new project-wide license is assigned by this
update.

The manuscript, compiled manuscript PDFs, original project README, handoff
notes, goals, operations records and credentials are excluded from this
repository. This guide describes the implementation and released data.
