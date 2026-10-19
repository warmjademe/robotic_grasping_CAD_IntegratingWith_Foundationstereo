"""Official GraspNet-Baseline, metric XYZ input; no camera hardware or robot I/O."""
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'upstream/graspnet-baseline'
PARAMETERS = dict(seed=0, num_points=20000, num_view=300, num_angle=12,
                  num_depth=4, cylinder_radius_m=.05, hmin_m=-.02,
                  hmax_list_m=[.01, .02, .03, .04], target_distance_m=.005,
                  collision_voxel_m=.01, collision_threshold=.01,
                  approach_distance_m=.05, collision_batch=64,
                  maximum_returned_candidates=200, nms=False, empty_filter=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_to_common(depths):
    """Align modeled distal fingertip planes; this is not physical TCP calibration."""
    assets = ROOT.parent/'baseline_graspgen/upstream/GraspGen/assets/franka'
    urdf, mesh = assets/'franka_panda.urdf', assets/'meshes/collision/finger.obj'
    joint = ET.parse(urdf).getroot().find("joint[@name='panda_finger_joint1']")
    mount = np.array(list(map(float, joint.find('origin').attrib['xyz'].split())))
    vertices = np.array([list(map(float, line.split()[1:])) for line in mesh.read_text().splitlines()
                         if line.startswith('v ')])
    assert np.allclose(mount[:2], 0)
    tip_z = float(mount[2] + vertices[:, 2].max())
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    result = np.tile(np.eye(4), (len(depths), 1, 1))
    result[:, :3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    result[:, 0, 3] = depths - tip_z
    evidence = dict(alignment='modeled_distal_fingertip_plane', panda_tip_z_m=tip_z,
                    panda_finger_joint_z_m=float(mount[2]),
                    panda_collision_mesh_local_zmax_m=float(vertices[:, 2].max()),
                    panda_urdf_sha256=sha(urdf), panda_finger_mesh_sha256=sha(mesh),
                    author_axes=dict(approach='+x', closing='+y', height='+z'),
                    author_origin='predicted surface seed; distal fingers end at local x=predicted depth',
                    formula='T_model_common.R=[[0,0,1],[0,1,0],[-1,0,0]]; t=[depth-panda_tip_z,0,0]',
                    physical_tcp_calibrated=False,
                    limitation='Author parallel-jaw collision geometry is not the Panda mesh; downstream Panda collision/TCP validation remains necessary')
    return result, evidence


def infer(arrays, metadata, workdir):
    """Return author poses in world coordinates, scores, decoded widths, record.

    ``record['T_model_to_common_grasp']`` is N x 4 x 4 and post-multiplies
    each returned author pose. Widths are never clipped to the 80 mm hand.
    """
    workdir = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT.parent/'shared'))
    from contract import validate
    validate(arrays, metadata)
    import torch
    from scipy.spatial import cKDTree
    from types import SimpleNamespace
    for p in [UPSTREAM/'models', UPSTREAM/'pointnet2', UPSTREAM/'utils', UPSTREAM/'knn']:
        sys.path.insert(0, str(p))
    from graspnet import GraspNet, pred_decode
    from collision_detector import ModelFreeCollisionDetector
    seed = int(metadata.get('inference_seed', PARAMETERS['seed']))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), 'This baseline requires verified CUDA custom operators'
    checkpoint = ROOT/'models/checkpoint-rs-mirror.tar'
    provenance = json.loads((ROOT/'models/provenance.json').read_text())
    assert sha(checkpoint) == provenance['sha256']
    scene = np.asarray(arrays['scene_points_m'], dtype=np.float32)
    target = np.asarray(arrays['target_points_m'], dtype=np.float32)
    T = np.asarray(arrays['T_world_reference'], dtype=np.float64)
    scene_ref = ((scene-T[:3, 3]) @ T[:3, :3]).astype(np.float32)
    target_ref = ((target-T[:3, 3]) @ T[:3, :3]).astype(np.float32)
    n = PARAMETERS['num_points']
    if len(scene_ref) >= n:
        indices = np.random.choice(len(scene_ref), n, replace=False)
    else:
        indices = np.concatenate([np.arange(len(scene_ref)), np.random.choice(len(scene_ref), n-len(scene_ref), replace=True)])
    model = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                    cylinder_radius=.05, hmin=-.02, hmax_list=[.01, .02, .03, .04], is_training=False).cuda().eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    restored = model.load_state_dict(state['model_state_dict'], strict=True)
    assert not restored.missing_keys and not restored.unexpected_keys
    inputs = {'point_clouds': torch.from_numpy(scene_ref[indices][None]).cuda()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        endpoints = model(inputs)
        decoded = pred_decode(endpoints)[0].cpu().numpy()
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter()-started
    assert decoded.ndim == 2 and decoded.shape[1] == 17 and np.isfinite(decoded).all()
    post_start = time.perf_counter()
    raw_n = len(decoded)
    target_distances = cKDTree(target_ref).query(decoded[:, 13:16])[0] if raw_n else np.empty(0)
    target_mask = target_distances < PARAMETERS['target_distance_m']
    target_ids = np.flatnonzero(target_mask)
    collision = np.zeros(raw_n, dtype=bool)
    empty = np.zeros(raw_n, dtype=bool)
    detector = ModelFreeCollisionDetector(scene_ref, voxel_size=PARAMETERS['collision_voxel_m'])
    for offset in range(0, len(target_ids), PARAMETERS['collision_batch']):
        ids = target_ids[offset:offset+PARAMETERS['collision_batch']]
        g = decoded[ids]
        group = SimpleNamespace(translations=g[:, 13:16], rotation_matrices=g[:, 4:13].reshape(-1, 3, 3),
                                heights=g[:, 2], depths=g[:, 3], widths=g[:, 1])
        coll, emp = detector.detect(group, approach_dist=PARAMETERS['approach_distance_m'],
                                   collision_thresh=PARAMETERS['collision_threshold'], return_empty_grasp=True)
        collision[ids], empty[ids] = coll, emp
    surviving_ids = np.flatnonzero(target_mask & ~collision)
    order = np.argsort(-decoded[surviving_ids, 0], kind='stable')
    selected = surviving_ids[order[:PARAMETERS['maximum_returned_candidates']]]
    g = decoded[selected]
    poses_ref = np.tile(np.eye(4), (len(g), 1, 1))
    poses_ref[:, :3, :3] = g[:, 4:13].reshape(-1, 3, 3)
    poses_ref[:, :3, 3] = g[:, 13:16]
    poses = T[None] @ poses_ref
    widths, depths, scores = g[:, 1].copy(), g[:, 3].copy(), g[:, 0].copy()
    common, alignment = model_to_common(depths)
    extra_path = workdir/'raw_extra.npz'
    if extra_path.exists():
        raise FileExistsError('Do not overwrite historical inference evidence')
    np.savez_compressed(extra_path, raw_grasp_rows=decoded, target_distances_m=target_distances,
                        target_mask=target_mask, collision_mask=collision, empty_grasp_mask=empty,
                        selected_raw_indices=selected, sampled_scene_indices=indices,
                        predicted_depth_m=depths, predicted_height_m=g[:, 2],
                        decoded_jaw_width_m=widths, T_model_to_common_grasp=common,
                        network_width_before_author_decode=endpoints['grasp_width_pred'].cpu().numpy())
    record = dict(method='GraspNet-Baseline', source='Official author network, decoding and model-free collision implementation',
                  upstream_commit='280c215129f759ed8649cb4e89fc5dfee55f4f80',
                  checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  checkpoint_provenance=provenance, official_checkpoint_verified=False,
                  strict_checkpoint_restore=True, restored_tensor_count=len(state['model_state_dict']),
                  checkpoint_epoch=int(state.get('epoch', -1)), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), parameters=dict(PARAMETERS, seed=seed),
                  seconds=forward_seconds, gpu_forward_seconds=forward_seconds,
                  postprocess_seconds=time.perf_counter()-post_start,
                  input_scene_points=len(scene_ref), input_target_points=len(target_ref),
                  raw_candidates=raw_n, target_filtered_candidates=int(target_mask.sum()),
                  collision_rejected=int(collision.sum()), collision_free_candidates=len(surviving_ids),
                  empty_flagged_target_candidates=int(empty[target_ids].sum()),
                  returned=len(g), candidate_budget=PARAMETERS['maximum_returned_candidates'],
                  width_source='network_prediction',
                  width_decode='Author pred_decode scales raw width by 1.2 and clamps to GRASP_MAX_WIDTH=0.1m; adapter does not clip to Panda 0.08m',
                  over_80mm_returned=int((widths > .08+1e-6).sum()),
                  T_model_to_common_grasp=common.tolist(), common_grasp_origin='panda_hand',
                  geometry_alignment=alignment,
                  grasp_frame_evidence=['upstream/graspnet-baseline/utils/collision_detector.py: finger x in (depth-0.06,depth), closing y',
                                        'upstream/graspnetAPI/graspnetAPI/utils/utils.py: plot_gripper_pro_max distal x=depth',
                                        'GraspGen/assets/franka/franka_panda.urdf and collision/finger.obj'],
                  raw_extra_file='raw_extra.npz', raw_extra_sha256=sha(extra_path),
                  backend_dependency_sha256={'backend.py':sha(__file__), 'graspnet.py':sha(UPSTREAM/'models/graspnet.py'),
                                 'collision_detector.py':sha(UPSTREAM/'utils/collision_detector.py')},
                  scene_scope=metadata['scene_scope'], collision_scope='observed input scene only; generic author gripper',
                  input_purpose=metadata['purpose'], input_geometry_source=metadata['geometry_source'],
                  input_source_hashes=metadata['input_hashes'],
                  input_contains_renderer_depth=metadata['renderer_depth_used'],
                  input_contains_scene_object_pose=metadata['object_pose_used'],
                  physical_robot_used=False, native_robot_execution_performed=False,
                  inference_completed=True, status='candidates_generated' if len(g) else 'no_candidates')
    return poses, scores, widths, record
