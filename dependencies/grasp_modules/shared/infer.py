#!/usr/bin/env python3
"""Generate common grasp candidates from either perception route. No robot I/O."""
import argparse
import importlib.util
import json
import os
import random
from pathlib import Path
import sys
import numpy as np
from contract import load, sha

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = {
    '6dof_graspnet': 'baseline_runtime/.venv-tf',
    'contact_graspnet': 'baseline_runtime/.venv-tf',
    'graspgen': 'baseline_graspgen/.venv-infer',
    'gpd': 'baseline_gpd/.venv',
    'graspnet_baseline': 'baseline_graspnet_baseline/.venv',
}
METHOD_IDS = tuple(ENVIRONMENTS)
PLUGIN_FOLDERS = {'gpd': 'baseline_gpd', 'graspnet_baseline': 'baseline_graspnet_baseline'}


def plugin_inference(method, arrays, metadata, workdir):
    path = ROOT / PLUGIN_FOLDERS[method] / 'backend.py'
    spec = importlib.util.spec_from_file_location('grasp_backend_' + method, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    poses, scores, widths, record = module.infer(arrays, metadata, workdir)
    record['backend_source_sha256'] = {str(path.relative_to(ROOT)): sha(path)}
    return poses, scores, widths, record


def validate_pose_batch(poses, count, label):
    poses = np.asarray(poses)
    if count == 0:
        poses = poses.reshape(0, 4, 4)
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(label + ' must contain finite N x 4 x 4 transforms')
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(label + ' has an invalid homogeneous row')
    rotation = poses[:, :3, :3]
    if not (np.allclose(np.linalg.det(rotation), 1, atol=1e-3)
            and np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-3)):
        raise ValueError(label + ' rotations must belong to SO(3)')
    return poses


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHOD_IDS, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=None,
                   help='Explicit repetition seed; omitted preserves historical backend defaults')
    args = p.parse_args()
    # Author 6-DOF code changes cwd. Resolve paths before loading that backend.
    args.input = args.input.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    environment = ROOT / ENVIRONMENTS[args.method]
    if Path(sys.prefix).resolve() != environment.resolve():
        exe = environment/'bin/python'
        if not exe.exists():
            raise RuntimeError('Required inference environment is not available at ' + str(environment))
        if args.method in ('6dof_graspnet', 'contact_graspnet'):
            libs = list((environment/'lib').glob('python*/site-packages/nvidia/*/lib'))
            os.environ['LD_LIBRARY_PATH'] = ':'.join(map(str, libs))+':'+os.environ.get('LD_LIBRARY_PATH', '')
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.execv(str(exe), [str(exe), str(Path(__file__).resolve()), *sys.argv[1:]])
    if args.out.exists():
        raise FileExistsError('Use a new output directory; historical results are immutable')
    arrays, metadata = load(args.input)
    args.out.mkdir(parents=True)
    seed = 0 if args.seed is None else args.seed
    if not 0 <= seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    np.random.seed(seed)
    random.seed(seed)
    if args.seed is not None:
        metadata = dict(metadata, inference_seed=seed)
    from backends import sixdof, contact, graspgen
    points = arrays['target_points_m']
    reference = arrays['T_world_reference']
    if args.method == '6dof_graspnet':
        poses, scores, widths, record = sixdof(points, reference, seed=seed)
    elif args.method == 'contact_graspnet':
        poses, scores, widths, record = contact(points, reference, arrays['scene_points_m'], seed=seed)
    elif args.method == 'graspgen':
        poses, scores, widths, record = graspgen(points, seed=seed)
    else:
        poses, scores, widths, record = plugin_inference(args.method, arrays, metadata, args.out)
    poses, scores, widths = np.asarray(poses), np.asarray(scores).reshape(-1), np.asarray(widths).reshape(-1)
    poses = validate_pose_batch(poses, len(scores), 'Author poses')
    if widths.shape != scores.shape or not np.isfinite(scores).all() or not np.isfinite(widths).all():
        raise ValueError('Scores and widths must be finite vectors with one value per grasp')
    # Same Panda hand convention as the historical FANUC adapters: canonical
    # grasp +z approaches, +y closes. Preserve the original author poses too.
    adapter = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    width_source = 'network_prediction' if args.method == 'contact_graspnet' else 'fixed_gripper_geometry'
    if args.method in PLUGIN_FOLDERS:
        # New author models have different axes AND origins. Require the
        # measured geometry mapping, including per-grasp depth where needed.
        adapter = np.asarray(record.pop('T_model_to_common_grasp'), dtype=np.float64)
        if adapter.shape != (4, 4):
            adapter = validate_pose_batch(adapter, len(scores), 'Model-to-common transforms')
        else:
            validate_pose_batch(adapter[None], 1, 'Model-to-common transform')
        if record.get('common_grasp_origin') != 'panda_hand' or not record.get('geometry_alignment'):
            raise ValueError('A documented author-gripper to Panda-hand geometry mapping is required')
        width_source = record['width_source']
        if width_source not in {'network_prediction', 'geometric_estimate', 'fixed_gripper_geometry'}:
            raise ValueError('Unsupported width provenance')
    world_grasp = poses @ adapter
    validate_pose_batch(world_grasp, len(scores), 'Common grasp poses')
    grasp_frame = dict(closing_axis='+y', approach_axis='+z', origin='panda_hand')
    if adapter.shape == (4, 4):
        grasp_frame['T_model_grasp'] = adapter.tolist()
    else:
        grasp_frame.update(T_model_grasp_field='T_model_grasp', transform_scope='per_candidate')
    width_valid = (widths > 0) & (widths <= metadata['gripper_geometry']['max_width_m']+1e-6)
    np.savez_compressed(args.out/'grasps.npz', T_world_grasp=world_grasp,
                        jaw_width_m=widths, score=scores, gripper_width_valid=width_valid,
                        model_poses=poses, T_model_grasp=adapter,
                        hand_poses=world_grasp, scores=scores, widths=widths)
    record.update(schema_version=1, requested_seed=args.seed,
                  effective_seed=20261011 if args.method == 'gpd' and args.seed is None else seed,
                  seed_scope={
                      'graspgen': ['Python random', 'NumPy global', 'PyTorch CPU/CUDA'],
                      '6dof_graspnet': ['Python random', 'NumPy global', 'TensorFlow graph'],
                      'contact_graspnet': ['Python random', 'NumPy global', 'TensorFlow graph'],
                      'graspnet_baseline': ['Python random', 'NumPy global', 'PyTorch CPU/CUDA'],
                      'gpd': ['NumPy Generator for target-point sampling'],
                  }[args.method],
                  bitwise_determinism_claim=False,
                  input_npz_sha256=sha(args.input),
                  input_json_sha256=sha(args.input.with_suffix('.json')),
                  input_source_hashes=metadata['input_hashes'],
                  perception_route=metadata.get('perception_route'),
                  scene_scope=metadata['scene_scope'], frame_id=metadata['frame_id'], units='m',
                  input_scene_points=len(arrays['scene_points_m']), input_target_points=len(points),
                  camera_origins_preserved=len(arrays['camera_origins_m']),
                  returned=len(poses), status='candidates_generated' if len(poses) else 'no_candidates',
                  inference_completed=True, gripper_id=metadata['gripper_id'],
                  width_source=width_source,
                  width_valid_count=int(width_valid.sum()),
                  grasp_frame=grasp_frame,
                  score_cross_model_calibrated=False,
                  downstream_pending=['width-aware planning', 'full-scene collision', 'TCP calibration for a different gripper'],
                  physical_robot_used=False, native_robot_execution_performed=False,
                  input_purpose=metadata['purpose'], input_geometry_source=metadata['geometry_source'],
                  input_contains_renderer_depth=metadata['renderer_depth_used'],
                  input_contains_scene_object_pose=metadata['object_pose_used'],
                  provenance_check='Validated declarations and source hashes; not automatic identification of arbitrary point-cloud origin',
                  source_sha256={f:sha(Path(__file__).with_name(f)) for f in ['infer.py','backends.py','contract.py']},
                  output_sha256=sha(args.out/'grasps.npz'))
    (args.out/'grasp_generation.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
