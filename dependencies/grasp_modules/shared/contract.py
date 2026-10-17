"""Versioned, perception-independent metric point-cloud grasp contract."""
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform_points(points, T):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def check_transform(T):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Expected a finite 4x4 rigid transform')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('Invalid homogeneous transform row')
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1, atol=1e-4):
        raise ValueError('Rotation must belong to SO(3)')
    return T


def validate(arrays, metadata):
    if metadata.get('schema_version') != VERSION or metadata.get('units') != 'm':
        raise ValueError('Input schema_version=1 and units=m are required')
    if not metadata.get('frame_id') or not metadata.get('input_hashes'):
        raise ValueError('Frame ID and source hashes are required')
    if metadata.get('purpose') != 'grasp_inference' or metadata.get('geometry_source') != 'perception_prediction':
        raise ValueError('Normal inference requires declared perception predictions, not evaluation-only geometry')
    if metadata.get('renderer_depth_used') is not False or metadata.get('object_pose_used') is not False:
        raise ValueError('Renderer depth/object-pose geometry is excluded from normal inference')
    if metadata.get('reference_convention') != 'opencv_optical_x_right_y_down_z_forward':
        raise ValueError('Declare the reference optical coordinate convention')
    if metadata.get('scene_scope') not in ['full_observed_scene', 'target_only', 'workspace_roi']:
        raise ValueError('Scene coverage must be declared explicitly')
    if metadata.get('gripper_id') != 'franka_panda':
        raise ValueError('Current verified backends require the Panda gripper checkpoint/geometry')
    geometry = metadata.get('gripper_geometry', {})
    if not np.isclose(geometry.get('max_width_m', -1), .08):
        raise ValueError('Current gripper uses an 80 mm maximum opening')
    for field, expected in [('closing_axis', '+y'), ('approach_axis', '+z'), ('tcp_frame', 'panda_hand')]:
        if geometry.get(field) != expected:
            raise ValueError('Inconsistent Panda gripper geometry: ' + field)
    for name in ['scene_points_m', 'target_points_m', 'camera_origins_m']:
        a = np.asarray(arrays[name])
        if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
            raise ValueError(name + ' must be a nonempty finite Nx3 array')
    if min(len(arrays['scene_points_m']), len(arrays['target_points_m'])) <= 128:
        raise ValueError('Current models require more than 128 points')
    check_transform(arrays['T_world_reference'])
    if 'point_camera_ids' in arrays:
        ids = np.asarray(arrays['point_camera_ids'])
        if ids.shape != (len(arrays['scene_points_m']),) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('point_camera_ids must be one integer per scene point')
        if np.any(ids < -1) or np.any(ids >= len(arrays['camera_origins_m'])):
            raise ValueError('Camera ID out of range; -1 means mixed/unknown source')
    if metadata.get('has_normals') != ('normals_world' in arrays):
        raise ValueError('Normal availability declaration does not match input')
    if metadata.get('has_colors') != ('scene_colors' in arrays):
        raise ValueError('Color availability declaration does not match input')
    return arrays, metadata


def load(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata.get('npz_sha256') != sha(path):
        raise ValueError('Input NPZ hash mismatch')
    return validate(arrays, metadata)


def save(path, arrays, metadata):
    path = Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('Preserve existing input artifacts')
    validate(arrays, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    metadata = dict(metadata, npz_sha256=sha(path))
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    return path
