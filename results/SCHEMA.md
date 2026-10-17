# Released experiment data

This directory contains the final recorded outcomes of the frozen single-seed
comparison and the completed source-ablation execution. The application is
Franka Emika Panda grasping in Isaac Sim and MuJoCo. The perception method fuses
CAD geometry with a stereo observation; grasp success is one downstream measure.

## Verify and read the results

Run `python3 results/verify_release.py` from the repository root. It uses only the
Python standard library and does not start experiments. It verifies all registered
outcome keys, recomputes object and scene success, checks exact paired tests and
Benjamini–Hochberg adjusted values, checks the reused outcomes, and validates
released file hashes. Its JSON output includes the recomputed group counts.

| Path | Contents |
| --- | --- |
| `comparison_seed0_v1/outcomes/{tless,itodd}.jsonl` | All 16,704 primary-comparison terminal outcomes, one JSON object per line. |
| `rq3_source_ablation_seed0_v1/outcomes/{tless,itodd}.jsonl` | 5,568 ablation-analysis outcomes: 2,784 new outcomes plus 2,784 outcomes reused from the primary comparison. |
| `<study>/summary.json` | Original object/scene aggregates, paired comparisons and review scope. |
| `<study>/geometry.jsonl` | Per-scene point-cloud evaluation; independent of simulator platform. |
| `comparison_seed0_v1/reports/*.csv` | Main-comparison method, condition, object, geometry, paired-test and measured cost exports. |
| `<study>/batch_audits/*.json` | Existing numerical checks of saved physical trajectories and calibration cases. |
| `<study>/source_inventory.jsonl.gz` | Gzip-compressed JSON Lines containing source-relative raw evidence identifiers and SHA-256 hashes. |
| `review_status.json` | What was audited, when it was audited, and what remains outside this release review. |
| `../data/release_provenance.json` | Original source-file hash and released-file hash for each transformed data artifact. |

Do not add the two study denominators together as independent trials. There are
19,488 unique task outcomes across the two studies. Each object has 12 fixed
scenes, and each scene/method/platform is evaluated once with seed 0. Old
three-seed studies and development controls are excluded; references to old
source paths occur only where an audited seed-0 outcome was reused.

## Outcome fields and denominators

`scene_id`, `object_id`, `dataset`, `condition`, `method`, `platform`, and `seed`
identify a registered evaluation. `task_id` identifies the corresponding terminal
record. `success` is the full physical-task criterion, not candidate generation,
finding a plan, or lifting alone. `executed` indicates whether a physical path was
executed. A completed no-path outcome has `success: false` and `executed: false`
and stays in the denominator. `planning_status`, `generation_status` and
`physical_metrics` preserve the separate stages and the measured failure modes.

The primary object success rate is
`number of objects with at least one successful scene / all dataset objects`.
It is computed separately for each method and platform: T-LESS has 30 objects,
ITODD has 28. Scene success rate is `successful outcomes / all registered scenes`:
360 for T-LESS and 336 for ITODD in each method/platform group. Both measures are
retained, including methods that perform worse than the fused point cloud.

`physical_metrics` records endpoint translation error in mm, rotation error in
degrees, bottom clearance in metres, hold duration in seconds and contact force
in newtons. The frozen full-success thresholds are endpoint error no larger than
2 mm / 0.5 degrees, bottom lift at least 0.1 m for 2 seconds, and prohibited
contact force no larger than 0.5 N, including the recorded return phase. The
runtime evaluator and frozen configuration determine the complete predicate.

`evidence`, `plan`, `trace`, `candidate_receipt`, `perception_record` and related
SHA-256 fields identify original archive evidence. These source-relative paths
are identifiers, not promises that the large files are bundled in this checkout.
The `formal_result` field distinguishes the original primary protocol from the
exploratory source ablation. `reused` is an accounting flag, not a new execution.
Shared physics-report timing must not be summed once per task; use the recorded
timing-scope field and deduplicate shared reports when aggregating execution cost.

## Point-cloud metrics and scene metadata

`geometry.jsonl` contains `surface_mae_mm`, `surface_p95_mm`, `precision_2mm` and
`complete_coverage_2mm`, with point-cloud and reference-surface hashes. Distance
fields use millimetres, and precision/coverage use proportions in [0,1]. The
primary file has 5,568 entries (8 perception configurations × 696 scenes). The
ablation file has 1,392 entries for the two new perception configurations; use
the primary file for the two reused configurations. Failed source gates remain
in the experiment denominator and should be interpreted through the gate status
and frozen evaluator, not silently removed.

`data/scene_index.json` lists all 696 scenes. Each
`data/scenes/<scene_id>/scene_spec.json` contains the simulation layout used to
render input images; it is ground truth for rendering/evaluation, not an online
pose input. `task.json` contains the fixed task region, and `cameras.json` records
calibrated camera geometry. `canonical_catalog.json` preserves original scale,
source hashes and the known support transform. Catalog `datasets/...` paths are
relative to the reproduction workspace created by the setup code.

## Review status and omitted large assets

The primary comparison has an existing passed evidence audit over 72,767 source
files. This release independently checks the released statistics and file
lineage. The source-ablation execution is complete and all 29 batch numerical
audits passed; an additional read-only hash check verified 6,720 raw result,
trajectory, calibration and planning/perception/candidate receipt files.
The final manuscript analysis of this ablation remains pending. These checks
are not a fresh physical replay or a claim that fusion wins every metric.

The release excludes the roughly 45 GB primary/ablation working directories,
raw trajectories, synthetic images, point-cloud arrays, weights and original
or derived CAD meshes. Obtain the pinned input archives from
`data/dataset_sources.json` and regenerate assets using the implementation.
SHA-256 inventories identify the retained original evidence, but there is no
public download endpoint for the omitted original execution traces in this
release. Regeneration may vary with simulator, driver and dependency versions;
it is not promised to be bitwise identical.

T-LESS uses CC BY 4.0 and MVTec ITODD uses CC BY-NC-SA 4.0. Their respective source
licenses and attribution requirements remain applicable; a code license does
not relicense third-party data. See `data/third_party_data.json` for official
source and license links.
