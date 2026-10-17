# Reproducing the final experiment

The release contains the final `adaptive_v6_thin_support` method, the 12-configuration
seed-zero comparison, and the registered-CAD/direct-concatenation source ablations.
The `runtime` directory is one deduplicated snapshot of the code shared by the
completed comparison batches. The archived numerical fusion, common planning,
physics, geometry metrics, and fixed-denominator success definitions are retained.
Path configuration and fresh-run preparation are release adapters; their hashes
and differences are recorded in `configs/code_provenance.json`.

## Recompute published tables without a GPU

Run `python3 results/verify_release.py`. It uses only Python's standard library and
checks every exported terminal outcome and the object/scene success denominators.
`code/single_run_metrics.py` and `code/coverage_metrics.py` contain the original
metrics implementation. Per-object success means at least one success in that
object's 12 registered scenes; unsuccessful and no-path attempts remain in the
scene denominator. The two simulators are reported separately.

## External source and model dependencies

Review `configs/upstream_sources.json` and its linked license files. Upstream
software, pretrained model weights, CAD meshes, and robot model assets are not
bundled. `python3 code/dependencies.py` prints the pinned source inventory;
`--fetch-sources` explicitly clones the listed repositories at their recorded
revisions. The GPD compatibility patch is distributed with its BSD license;
apply it to the pinned GPD checkout before building that baseline.

Set paths from the release directory (an absolute checkout path is required):

```sh
export CAD_RELEASE_ROOT="$PWD"
export CAD_WORKSPACE="$PWD/workspace"
export GRASP_MODULE_ROOT="$PWD/dependencies/grasp_modules"
export STEREO_MODULE_ROOT="$PWD/dependencies/stereo_modules"
export GRASPGEN_RUNTIME="$GRASP_MODULE_ROOT/baseline_graspgen"
export CAD_GPU_LOCK="$CAD_WORKSPACE/gpu.lock"
```

Install the author dependencies in separate environments, following their pinned
installation instructions. The observed versions are in `configs/environments.json`:
GraspGen/perception use `.venv-infer`, MuJoCo 3.3.7 uses `.venv-mujoco`, and Isaac
Sim 5.1.0.0 / Isaac Lab 2.3.2.post1 use `.venv-sim`, all below `GRASPGEN_RUNTIME`.
The alternative grasp adapters select the environments listed in
`dependencies/grasp_modules/shared/infer.py`. This package record is not a complete
pip or system-library lock; CUDA compilation and the simulators require Linux and
a compatible NVIDIA GPU. The measured experiments used an RTX 4090 (24 GiB).

Restore the author checkpoints to the paths listed in `configs/model_weights.json`
and run `python3 code/dependencies.py --check-weights`. Official download instructions
are linked from each pinned upstream repository. GraspGen used
`adithyamurali/GraspGenModels` revision `ec1ccbb5eec0680db669246ac312a3636f16ee43`.
The GraspNet-Baseline checkpoint used a third-party mirror; its precise provenance
is in `dependencies/grasp_modules/baseline_graspnet_baseline/models/provenance.json`.
Its identity with the inaccessible official checkpoint has not been verified.
Model licenses, including noncommercial restrictions, apply independently of this
release. No new license is asserted for third-party material.

The compatibility build helpers under `dependencies/grasp_modules` retain the
original TensorFlow 2.15.1 and RTX 4090 compilation settings. Set `CUDA_HOME` before
`baseline_runtime/build_tf_ops.py`, run `validate_tf_ops.py`, then
`prepare_6dof_compat.py`. Build GPD with its `build.sh` after installing its
Eigen/PCL/OpenCV toolchain. Build GraspNet-Baseline with `build_ops.sh` and verify
its custom operators with `validate_cuda_ops.py`. No old experiment or dashboard
scripts are needed. Downloading sources alone does not install these environments.

## Restore CAD and prepare fresh scenes

Acquire the T-LESS and ITODD models from `data/dataset_sources.json` under the
datasets' own terms. Unpack an `original` directory with `tless/models_cad/` and
`itodd/models/`. With NumPy 1.26.4 and trimesh 4.5.3 available:

```sh
python3 code/prepare_assets.py --original /path/to/original --workspace "$CAD_WORKSPACE"
python3 code/prepare_experiment.py --workspace "$CAD_WORKSPACE" \
  --menagerie "$PWD/dependencies/assets/menagerie" --batch 0
```

`prepare_assets.py` verifies the original mesh hashes, millimetre-to-metre conversion,
the CAD-only support transformation, and the exact canonical mesh/sample hashes.
It preserves shape and scale. `prepare_experiment.py` copies the archived scene
specifications and task definitions into a new working directory; the 696 scene
identities and 29 batch assignments are fixed in `data` and `configs`.
Rendering produces new RGB images and evaluation-only ground truth. The archived
camera JSON files provide the original calibration for comparison.

## Execute and audit a batch

```sh
python3 code/run_experiment.py --run "$CAD_WORKSPACE/comparison_seed0_v1/batches/batch_000" --list
python3 code/run_experiment.py --run "$CAD_WORKSPACE/comparison_seed0_v1/batches/batch_000"
```

The listed stages calibrate both simulators; render and freeze inputs; run stereo,
CAD registration and fusion; generate grasps; plan; execute physics; and audit
the raw trajectories. Each scene/method executes once with seed zero. Perception
and grasp inference run in a Linux network namespace using `sudo -n unshare`,
`setpriv`, and the original offline entry point. This requires an explicitly
configured local privilege policy. Install all assets before inference.
The wrapper does not enable sudo access. Install and use each simulator under
its applicable terms before running its existing headless launch scripts.

For another batch, prepare a new batch index (0 through 28). Existing preparations
and stage receipts are preserved and rejected rather than reset. Use `--stage NAME`
only for a deliberately selected stage after its prerequisites have completed;
the runner does not diagnose or automatically retry interrupted attempts.
Every new execution belongs to `workspace`, separate from published `results`.

After auditing all main-comparison batches, aggregate with:

```sh
python3 code/single_run_metrics.py aggregate --study "$CAD_WORKSPACE/comparison_seed0_v1" --root "$CAD_WORKSPACE"
```

Then prepare the two source
ablations with the same command plus `--source-ablations`, then run the resulting
`rq3_source_ablation_seed0_v1/batches/batch_000` directory. The archived v6 estimated
pose and acceptance gate are reused. Registered-CAD-only still uses stereo to
estimate alignment and scene obstacles. Rejected registration gates stay as
failed scheduled tasks rather than being dropped or silently replaced by stereo.
Aggregate the completed ablation batches with
`python3 code/rq3_ablation_metrics.py aggregate --study "$CAD_WORKSPACE/rq3_source_ablation_seed0_v1"`.

## Verification scope

The original study's results were generated and audited before packaging.
This release was checked by parsing the source, preserving numerical-core hashes,
reconstructing CAD assets, preparing new batch directories, listing execution
commands, and recomputing exported outcome statistics. GPU inference and physics
were not rerun merely to publish the package. A fresh end-to-end reproduction
requires the external dependencies, environments, licenses, and compute above.
