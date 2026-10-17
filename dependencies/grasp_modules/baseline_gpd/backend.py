"""Author GPD/Eigen adapter. Returns author poses; caller applies recorded SE(3)."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
COMMIT = '6327f20eabfcba41a05fdd2e2ba408153dc2e958'
TIP_FROM_PANDA_HAND_M = .112249
HAND_DEPTH_M = .06


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_to_common():
    # T_author_common: common +z=author+x, common+y=author+y.
    # Author base-to-tip=.06; Panda hand-to-tip=.0584+.053849.
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    T[0, 3] = HAND_DEPTH_M - TIP_FROM_PANDA_HAND_M
    return T


def verify_author_files():
    manifest=json.loads((ROOT/'model_manifest.json').read_text())
    actual_commit=subprocess.check_output(['git','-C',str(ROOT/'upstream/gpd'),'rev-parse','HEAD'],text=True).strip()
    if actual_commit != COMMIT or manifest['commit'] != COMMIT:
        raise ValueError('GPD upstream commit differs from frozen official snapshot')
    counts={'conv1_weights.bin':20*15*5*5,'conv1_biases.bin':20,'conv2_weights.bin':50*20*5*5,'conv2_biases.bin':50,'ip1_weights.bin':500*50*12*12,'ip1_biases.bin':500,'ip2_weights.bin':2*500,'ip2_biases.bin':2}
    for name,count in counts.items():
        relative='upstream/gpd/models/lenet/15channels/params/'+name
        p=ROOT/relative
        spec=manifest['files'][relative]
        if not p.is_file() or p.stat().st_size != count*4 or sha(p) != spec['sha256']:
            raise ValueError('Missing, truncated or changed official weights: '+name)
        if not np.isfinite(np.fromfile(p,dtype='<f4')).all():
            raise ValueError('Nonfinite official weights: '+name)
    geometry=(ROOT/'upstream/gpd/cfg/hand_geometry.cfg').read_text()
    parsed={line.split('=',1)[0].strip():line.split('=',1)[1].strip() for line in geometry.splitlines() if '=' in line and not line.lstrip().startswith('#')}
    if float(parsed['hand_depth']) != HAND_DEPTH_M:
        raise ValueError('Author hand depth differs from registered frame transform')
    return manifest


def visibility_for_gpd(points, origins, ids):
    """Keep measured provenance; estimate geometric visibility only for unknowns.

    Spherical flipping/convex hull (hidden point removal), radius 100*max
    point-camera distance. This is an input adapter, not measured provenance.
    Unknown points invisible in every hull are excluded, never assigned a fake
    camera. Input files and their original -1 IDs are kept unchanged.
    """
    from scipy.spatial import ConvexHull
    visibility = np.zeros((len(points), len(origins)), dtype=np.int32)
    known = ids >= 0
    visibility[np.flatnonzero(known), ids[known]] = 1
    unknown = ~known
    if unknown.any():
        for j, origin in enumerate(origins):
            relative = points - origin
            norm = np.linalg.norm(relative, axis=1)
            if np.any(norm <= 1e-10):
                raise ValueError('Camera origin coincides with input point')
            radius = 100.0 * norm.max()
            flipped = relative * ((2.0 * radius / norm) - 1.0)[:, None]
            hull = ConvexHull(np.vstack([flipped, np.zeros((1, 3))]))
            visible = hull.vertices[hull.vertices < len(points)]
            visible = visible[unknown[visible]]
            visibility[visible, j] = 1
    retain = visibility.any(axis=1)
    if retain.sum() < 129:
        raise ValueError('Insufficient points with measured or estimated visibility')
    return visibility, retain, {
        'input_camera_ids_preserved': True,
        'measured_source_points': int(known.sum()),
        'unknown_source_points': int(unknown.sum()),
        'estimated_visibility_method': 'spherical_flipping_convex_hull_radius_100x_max_distance' if unknown.any() else None,
        'estimated_visibility_is_observed_provenance': False,
        'camera_point_counts': visibility.sum(axis=0).tolist(),
        'excluded_without_visible_camera': int((~retain).sum()),
        'normal_orientation': 'author PCL radius normals towards first listed measured/estimated visible camera; author reverseNormals checks all listed cameras',
        'shadow_convention': 'author multi-camera intersection using measured sources or explicitly estimated visibility',
    }


def infer(arrays, metadata, workdir):
    from sys import path
    path.insert(0, str(ROOT.parent / 'shared'))
    from contract import validate
    validate(arrays, metadata)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if any(workdir.iterdir()):
        raise FileExistsError('GPD workdir must be empty; preserve historical attempts')
    binary = ROOT / 'build/gpd_shared_bridge'
    if not binary.exists():
        raise FileNotFoundError('Build official GPD with build.sh first')
    verify_author_files()
    start = time.perf_counter()
    points = np.asarray(arrays['scene_points_m'], dtype=np.float64)
    target = np.asarray(arrays['target_points_m'], dtype=np.float64)
    origins = np.asarray(arrays['camera_origins_m'], dtype=np.float64)
    ids = np.asarray(arrays.get('point_camera_ids', np.full(len(points), -1)), dtype=int)
    visibility, retain, provenance = visibility_for_gpd(points, origins, ids)
    points = points[retain]
    visibility = visibility[retain]
    # Fixed target-only sampling; all retained scene points remain collision input.
    seed = int(metadata.get('inference_seed', 20261011))
    rng = np.random.default_rng(seed)
    samples = target[rng.choice(len(target), min(200, len(target)), replace=False)]
    input_path = workdir / 'bridge_input.txt'
    with input_path.open('w') as stream:
        stream.write(f'{len(points)} {len(origins)} {len(samples)}\n')
        np.savetxt(stream, origins, fmt='%.17g')
        np.savetxt(stream, np.column_stack([points, visibility]), fmt=['%.17g']*3+['%d']*len(origins))
        np.savetxt(stream, samples, fmt='%.17g')
    np.savez_compressed(workdir / 'visibility_adapter.npz', retained_input_indices=np.flatnonzero(retain), camera_visibility=visibility, target_samples_m=samples)
    author = ROOT / 'upstream/gpd'
    cfg = (author / 'cfg/eigen_params.cfg').read_text()
    values = {
        'hand_geometry_filename': str(author / 'cfg/hand_geometry.cfg'),
        'image_geometry_filename': str(author / 'cfg/image_geometry_15channels.cfg'),
        'weights_file': str(author / 'models/lenet/15channels/params')+'/',
        'num_samples': '200', 'num_threads': '1', 'num_selected': '200',
        'min_aperture': '0', 'max_aperture': '0.08',
        'workspace': ' '.join(str(v) for pair in zip(points.min(0)-.15, points.max(0)+.15) for v in pair),
        'workspace_grasps': ' '.join(str(v) for pair in zip(points.min(0)-.15, points.max(0)+.15) for v in pair),
    }
    lines=[]
    for line in cfg.splitlines():
        key=line.split('=',1)[0].strip()
        if key.startswith('plot_') and '=' in line:
            line=key+' = 0'
        elif key in values:
            line=key+' = '+values.pop(key)
        lines.append(line)
    lines.extend(k+' = '+v for k,v in values.items())
    config_path=workdir/'gpd.cfg'
    config_path.write_text('\n'.join(lines)+'\n')
    env=os.environ.copy()
    # Author Eigen classifier shares scratch vectors across images. Running one
    # OpenMP thread avoids races without altering any layer or learned weight.
    env.update(OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1', OPENBLAS_NUM_THREADS='1')
    env_prefix=ROOT/'tooling/mamba-root/envs/env'
    env['LD_LIBRARY_PATH']=str(env_prefix/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    command=[str(binary), str(config_path), str(input_path), str(workdir/'author_grasps.txt')]
    infer_start=time.perf_counter()
    with (workdir/'author_inference.log').open('w') as stream:
        process=subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=1800)
    elapsed=time.perf_counter()-infer_start
    attempt={'command':command,'returncode':process.returncode,'inference_wall_seconds':elapsed,'visibility':provenance}
    (workdir/'attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
    if process.returncode:
        raise RuntimeError('Author GPD failed; see preserved author_inference.log')
    raw_file=workdir/'author_grasps.txt'
    rows=np.loadtxt(raw_file, ndmin=2) if raw_file.stat().st_size else np.empty((0,16))
    poses=np.tile(np.eye(4),(len(rows),1,1))
    if len(rows): poses[:,:3,:]=rows[:,:12].reshape(-1,3,4)
    scores=rows[:,12]
    widths=rows[:,13]
    if not np.isfinite(rows).all():
        raise ValueError('Author returned nonfinite candidates; preserved raw output')
    if len(rows) and not np.allclose(rows[:,15]-rows[:,14],HAND_DEPTH_M,atol=1e-8):
        raise ValueError('Author output closing-box depth violates registered transform')
    if len(rows) and (not np.allclose(poses[:,:3,:3].transpose(0,2,1)@poses[:,:3,:3],np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-5)):
        raise ValueError('Author returned nonrigid rotations; preserved raw output')
    valid=(widths>0)&(widths<=.08)
    weights={str(p.relative_to(ROOT)):sha(p) for p in sorted((author/'models/lenet/15channels/params').glob('*.bin'))}
    record={
        'backend':'gpd','official_commit':COMMIT,'device':'cpu','classifier':'official Eigen 15-channel LeNet',
        'checkpoint_files_sha256':weights,'binary_sha256':sha(binary),'config_sha256':sha(config_path),
        'width_source':'geometric_estimate',
        'width_semantics':'observed_object_span_in_author_closing_region',
        'width_is_network_prediction':False,'author_nominal_max_opening_m':.10,
        'width_filter_max_m':.08,'model_geometry_is_panda_geometry':False,
        'T_model_to_common_grasp':model_to_common().tolist(),
        'common_grasp_origin':'panda_hand','raw_pose_origin':'author_closing_box_base_center',
        'raw_axes':'+x approach; +y closing/binormal; +z hand axis',
        'geometry_alignment':{'method':'align_fingertip_plane_and_closing_midplane','author_base_to_tip_m':HAND_DEPTH_M,'panda_hand_to_tip_m':TIP_FROM_PANDA_HAND_M,'translation_author_x_m':HAND_DEPTH_M-TIP_FROM_PANDA_HAND_M,'limitations':'Axis/tip-plane registration only; author rectangular 100mm hand differs from Panda mesh. Panda collision and width-aware planning remain required.'},
        'geometry_source_evidence':['upstream/gpd/src/gpd/candidate/hand.cpp:42','upstream/gpd/src/gpd/candidate/finger_hand.cpp:29','upstream/gpd/include/gpd/candidate/hand.h:226','upstream/gpd/cfg/hand_geometry.cfg','../baseline_graspgen/upstream/GraspGen/assets/franka/franka_panda.urdf'],
        'score_semantics':'author positive logit minus negative logit; not calibrated probability or cross-model common scale',
        'sampling':{'seed':seed,'target_samples':len(samples),'orientations':8,'finger_placements':10,'selected_limit':200,'voxel_size_m':.003},
        'visibility_adapter':provenance,'scene_scope':metadata['scene_scope'],
        'input_npz_sha256':metadata.get('npz_sha256'),'input_hashes':metadata['input_hashes'],
        'author_output_count':len(rows),'filtered_invalid_width_or_nonfinite':int((~valid).sum()),
        'inference_wall_seconds':elapsed,'adapter_total_seconds':time.perf_counter()-start,
        'gpu_inference':False,'robot_execution':False,'full_unified_planning_verified':False,
    }
    (workdir/'backend_record.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(workdir/'author_output.npz',T_world_author=poses,scores=scores,widths_m=widths,retained=valid)
    return poses[valid],scores[valid],widths[valid],record
