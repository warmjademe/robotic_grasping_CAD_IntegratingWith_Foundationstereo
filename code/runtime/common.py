"""Utilities for the isolated, known-CAD stereo grasping experiment."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

RUN = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ['CAD_RELEASE_ROOT']).resolve()
STUDY = RUN
RUNTIME = Path(os.environ['GRASPGEN_RUNTIME']).resolve()
GRASP_MODULES = Path(os.environ['GRASP_MODULE_ROOT']).resolve()
GPU_LOCK = Path(os.environ.get('CAD_GPU_LOCK', str(PROJECT/'workspace/gpu.lock')))
PROTOCOL = json.loads((RUN / 'protocols/protocol.json').read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def translation(xyz):
    T = np.eye(4)
    T[:3, 3] = xyz
    return T

def pose7(T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.r_[T[:3, 3], q[3], q[:3]]

def transform(points, T):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]

def voxel(points, size=.001):
    points = np.asarray(points)
    if not len(points):
        return points.reshape(-1, 3)
    _, indices = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(indices)]

@contextlib.contextmanager
def gpu_lock():
    import fcntl
    path = GPU_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def interpolate(path, fraction):
    path = np.asarray(path)
    if len(path) == 1:
        return path[0]
    x = np.clip(fraction, 0, 1) * (len(path) - 1)
    i = min(int(x), len(path) - 2)
    return path[i] * (1 - (x-i)) + path[i+1] * (x-i)

def command(selection, phase, fraction):
    paths = selection['paths']
    smooth = fraction * fraction * (3 - 2 * fraction)
    if phase == 'settle': q = paths['transit'][0]
    elif phase in ['transit', 'approach', 'lift']: q = interpolate(paths[phase], smooth)
    elif phase in ['approach_hold', 'close', 'release']: q = paths['approach'][-1]
    elif phase == 'hold': q = paths['lift'][-1]
    elif phase == 'lower': q = interpolate(paths['lift'], 1-smooth)
    elif phase == 'retreat': q = interpolate(paths['approach'], 1-smooth)
    else: q = paths['approach'][0]
    opening = selection['opening'] if phase in ['settle', 'transit', 'approach', 'approach_hold', 'release', 'retreat', 'placed_hold'] else selection['closing']
    return np.asarray(q), opening

def outcome(poses, phases, vertices, baseline, endpoint, bad_steps, max_arm, max_finger):
    poses = np.asarray(poses); phases = np.asarray(phases)
    rot = Rotation.from_quat(poses[:, [4,5,6,3]]).as_matrix()
    bottom = np.min(np.einsum('tv,pv->tp', rot[:,2,:], vertices), axis=1) + poses[:,2]
    c = PROTOCOL['physics']; hold = bottom[phases == 'hold']
    lift = bool(len(hold) == c['trace_hz'] * c['hold_seconds'] and np.min(hold) >= c['lift_bottom_clearance_m'])
    ep, er = endpoint
    reached = bool(ep <= c['endpoint_position_m'] and er <= c['endpoint_rotation_degrees'])
    placed = poses[phases == 'placed_hold']; placed_bottom = bottom[phases == 'placed_hold']
    xy = float(np.linalg.norm(placed[-1,:2] - np.asarray(baseline)[:2]))
    returned = bool(lift and np.all(np.abs(placed_bottom) < c['return_bottom_m']) and xy <= c['return_xy_m'])
    return dict(physics_lift_success=lift, approach_endpoint_reached=reached,
        collision_free=bad_steps == 0, execution_success=bool(lift and reached and bad_steps == 0),
        execution_and_return=bool(lift and reached and bad_steps == 0 and returned),
        returned_to_origin_success=returned, prohibited_contact_steps=int(bad_steps),
        max_arm_contact_force_n=float(max_arm), max_finger_non_target_force_n=float(max_finger),
        approach_position_error_mm=float(ep*1000), approach_rotation_error_deg=float(er),
        min_hold_bottom_clearance_m=float(hold.min()), return_xy_error_m=xy,
        final_bottom_clearance_m=float(placed_bottom[-1]), hold_sim_s=c['hold_seconds'],
        object_attached=False, object_pose_written_after_initialization=False)
