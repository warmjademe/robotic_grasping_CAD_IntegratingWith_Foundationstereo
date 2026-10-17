"""Confidence-aware upright CAD registration and two-view surface fusion.

Runtime inputs: RGB, calibration, predicted depth, observed cloud, known CAD.
Ground-truth/evaluation reads are explicitly rejected by the entrypoint.
"""
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from fusion_common import ROOT, SOURCE, read, write, sha


def transform(p, T):
    return p @ T[:3, :3].T + T[:3, 3]


def voxel_indices(points, size):
    if not len(points):
        return np.empty(0, dtype=int)
    _, ix = np.unique(np.floor(points/size).astype(np.int64), axis=0, return_index=True)
    return np.sort(ix)


def project(points, camera, shape):
    K = np.asarray(camera['K'])
    pc = transform(points, np.linalg.inv(np.asarray(camera['world_from_camera'])))
    z = pc[:, 2]
    uv = np.rint(pc[:, :2]/np.maximum(z[:, None], 1e-8)*K[[0, 1], [0, 1]] + K[:2, 2]).astype(int)
    h, w = shape
    inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, z, inside


def sample_map(points, camera, image, missing=0.):
    uv, _, inside = project(points, camera, image.shape[:2])
    out = np.full(len(points), missing, dtype=float)
    out[inside] = image[uv[inside, 1], uv[inside, 0]]
    return out


def stereo_confidence(disparity, other_disparity, image, other_image, direction, cfg):
    yy, xx = np.indices(disparity.shape, dtype=np.float32)
    x_other = xx + direction*disparity
    sampled = cv2.remap(other_disparity, x_other, yy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    rgb = cv2.remap(other_image.astype(np.float32), x_other, yy, cv2.INTER_LINEAR)
    residual = np.abs(disparity-sampled)
    photo = np.sqrt(np.mean((image.astype(float)-rgb)**2, axis=2))
    valid = np.isfinite(disparity) & np.isfinite(sampled) & (disparity > 0) & (sampled > 0)
    valid &= (x_other >= 0) & (x_other < image.shape[1]-1)
    q = np.exp(-.5*(residual/cfg['left_right_sigma_px'])**2)
    q *= np.exp(-.5*(photo/cfg['photometric_sigma_rgb'])**2)**.25
    q[~valid] = 0
    return q.astype(np.float32), residual.astype(np.float32), photo.astype(np.float32)


def foreground_confidence(image, camera, roi, table_z, cfg):
    # Estimate the support's appearance from a task-defined annulus, rather
    # than assuming a fixed object or table colour or reading a rendered mask.
    low, high = np.array(roi['min']), np.array(roi['max'])
    x = np.linspace(low[0]-.04, high[0]+.04, 64)
    y = np.linspace(low[1]-.025, high[1]+.025, 64)
    xx, yy = np.meshgrid(x, y)
    ring = (xx < low[0]-.005) | (xx > high[0]+.005) | (yy < low[1]-.005) | (yy > high[1]+.005)
    points = np.column_stack([xx[ring], yy[ring], np.full(ring.sum(), table_z)])
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(float)
    uv, _, inside = project(points, camera, image.shape[:2])
    uv = uv[inside]
    if len(uv) < 32:
        return np.ones(image.shape[:2], np.float32), {'available': False}
    samples = lab[uv[:, 1], uv[:, 0]]
    center = np.median(samples, axis=0)
    distance = np.linalg.norm(lab-center, axis=2)
    sample_distance = np.linalg.norm(samples-center, axis=1)
    median = np.median(sample_distance)
    noise = max(cfg['foreground_lab_noise_floor'], 1.4826*np.median(np.abs(sample_distance-median)))
    excess = np.maximum(distance-median-cfg['foreground_threshold_sigma']*noise, 0)
    q = 1-np.exp(-.5*(excess/noise)**2)
    return q.astype(np.float32), {'available': True, 'background_lab': center.tolist(),
        'noise_lab': float(noise), 'sample_count': len(samples)}


def visible_indices(points, camera, shape):
    uv, z, inside = project(points, camera, shape)
    ix = np.where(inside)[0]
    ix = ix[np.argsort(z[ix])]
    _, unique = np.unique(uv[ix, 1]*shape[1]+uv[ix, 0], return_index=True)
    return ix[unique]


def fit_cad(cad, observed, weights, bounds, cameras, fg, cfg):
    ix = voxel_indices(observed, cfg['voxel_m'])
    if len(ix) > cfg['maximum_observed_points']:
        ix = ix[np.linspace(0, len(ix)-1, cfg['maximum_observed_points']).astype(int)]
    obs, weights = observed[ix], weights[ix]
    cad_tree, obs_tree = cKDTree(cad), cKDTree(obs)
    height = bounds[1, 2]-bounds[0, 2]
    centers = [np.median(obs[:, :2], axis=0), np.average(obs[:, :2], axis=0, weights=weights)]
    high = obs[:, 2] >= max(.005, height*.45)
    if high.sum() >= 32:
        centers.append(np.median(obs[high, :2], axis=0))
    trials = []
    best = None
    for center in centers[:cfg['maximum_center_initializations']]:
        for yaw in np.linspace(-np.pi, np.pi, cfg['yaw_initializations'], endpoint=False):
            R = Rotation.from_euler('z', yaw).as_matrix()
            rot = cad @ R.T
            t = np.r_[center-(rot.min(0)+rot.max(0))[:2]/2, -bounds[0, 2]]
            for iteration in range(cfg['max_iterations']):
                d, nearest = cad_tree.query((obs-t) @ R)
                robust = weights/(1+(d/cfg['robust_distance_m'])**2)
                use = d < max(cfg['inlier_distance_m'], min(.02, np.quantile(d, .8)))
                if use.sum() < 30:
                    break
                w = robust[use]; w /= w.sum()
                a, b = cad[nearest[use], :2], obs[use, :2]
                ac, bc = np.sum(a*w[:, None], 0), np.sum(b*w[:, None], 0)
                u, _, vt = np.linalg.svd(((a-ac)*w[:, None]).T @ (b-bc))
                rr = vt.T @ u.T
                if np.linalg.det(rr) < 0:
                    vt[-1] *= -1
                    rr = vt.T @ u.T
                newR = np.eye(3); newR[:2, :2] = rr
                newt = t.copy(); newt[:2] = bc-rr @ ac
                change = np.linalg.norm(newt-t)+.03*np.linalg.norm(newR-R)
                R, t = newR, newt
                if change < 1e-6:
                    break
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
            fitted = transform(cad, T)
            d, _ = cad_tree.query((obs-t) @ R)
            forward = np.average(np.minimum(d, .008), weights=weights)
            reverse, silhouettes = [], []
            for side in ['left', 'right']:
                vi = visible_indices(fitted, cameras[side], fg[side].shape)
                vi = vi[fitted[vi, 2] > cfg['cad_surface_minimum_height_m']]
                reverse.append(float(np.minimum(obs_tree.query(fitted[vi])[0], .008).mean()) if len(vi) else .008)
                silhouettes.append(float(1-sample_map(fitted[vi], cameras[side], fg[side]).mean()) if len(vi) else 1.)
            score = float(forward+.5*np.mean(reverse)+.003*np.mean(silhouettes))
            overlap = float(np.average(d < cfg['inlier_distance_m'], weights=weights))
            record = dict(initial_center_xy=center.tolist(), initial_yaw_rad=float(yaw),
                score_m=score, overlap=overlap, estimated_world_from_cad=T.tolist(), iterations=iteration+1)
            trials.append(record)
            if best is None or score < best['score_m']:
                best = record
    q = float(best['overlap']*np.exp(-best['score_m']/.008))
    return np.array(best['estimated_world_from_cad']), q, best, trials


def process(source, reverse, output, config, sid):
    inp = source/'datasets/inputs'/sid
    base = source/'results/perception/foundationstereo'/sid
    cal, task = read(inp/'cameras.json'), read(inp/'task.json')
    camera = cal['cameras']
    meshpath = source/'datasets/assets'/task['object_id']/'mesh_m.ply'
    paths = [inp/'left.png', inp/'right.png', inp/'cameras.json', inp/'task.json',
             base/'cloud.npz', base/'prediction.npz', reverse/sid/'prediction.npz', meshpath]
    samples = source/'datasets/assets'/task['object_id']/'cad_samples.npz'
    if samples.exists():
        paths.append(samples)
    hashes = {str(p): sha(p) for p in paths}
    out = output/sid
    if (out/'record.json').exists():
        prior = read(out/'record.json')
        assert prior['input_sha256'] == hashes
        assert prior['config'] == config and prior['code_sha256'] == sha(__file__)
        return prior
    start = time.perf_counter()
    images = {s: cv2.cvtColor(cv2.imread(str(inp/f'{s}.png')), cv2.COLOR_BGR2RGB) for s in ['left', 'right']}
    with np.load(base/'prediction.npz') as d:
        left = {k: d[k] for k in d.files}
    with np.load(reverse/sid/'prediction.npz') as d:
        right = {k: d[k] for k in d.files}
    pred = {'left': left, 'right': right}
    qdepth, fg, foreground_records = {}, {}, {}
    for side, other, direction in [('left', 'right', -1), ('right', 'left', 1)]:
        qdepth[side], _, _ = stereo_confidence(pred[side]['disparity'], pred[other]['disparity'],
            images[side], images[other], direction, config['confidence'])
        fg[side], foreground_records[side] = foreground_confidence(images[side], camera[side],
            task['target_roi_world_m'], task['known_table_height_m'], config['confidence'])
    with np.load(base/'cloud.npz') as d:
        observed = d['target_points_m'].astype(float)
        original_scene = d['scene_points_m']
    mesh = trimesh.load(meshpath, force='mesh', process=False)
    if samples.exists():
        with np.load(samples) as d:
            cad = d['points_m']
    else:
        cad, _ = trimesh.sample.sample_surface(mesh, 20000, seed=int(task['object_id'].split('_')[-1]))
    qv = sample_map(observed, camera['left'], qdepth['left'])
    qfg = sample_map(observed, camera['left'], fg['left'])
    # The CAD height is an allowed known-model prior, with a 10 mm tolerance.
    eligible = (qfg >= config['confidence']['minimum_foreground_weight'])
    eligible &= observed[:, 2] <= np.ptp(mesh.vertices[:, 2])+.01
    clean = observed[eligible]
    weights = (.05+.95*qv[eligible])*qfg[eligible]
    registration = config['registration']; fusion = config['fusion']
    best, trials, T, qc = None, [], None, 0.
    if len(clean) >= registration['minimum_observations']:
        T, qc, best, trials = fit_cad(cad, clean, weights, mesh.bounds, camera, fg, registration)
    fallback = best is None or qc < registration['minimum_confidence'] or best['overlap'] < registration['minimum_overlap']
    blocked = {'left': 0, 'right': 0}
    if fallback:
        fused, origins, conf = observed, np.zeros(len(observed), np.uint8), qv
    else:
        registered = transform(cad, T)
        registered = registered[registered[:, 2] > .003]
        distances, nearest = cKDTree(registered).query(clean)
        alpha = qc*(1-qv[eligible])*np.exp(-.5*(distances/fusion['correspondence_sigma_m'])**2)
        alpha = np.clip(alpha, 0, fusion['max_cad_weight'])
        corrected = clean*(1-alpha[:, None])+registered[nearest]*alpha[:, None]
        usable = np.ones(len(registered), bool)
        for side in ['left', 'right']:
            uv, z, inside = project(registered, camera[side], pred[side]['depth'].shape)
            ix = np.where(inside)[0]
            measured = pred[side]['depth'][uv[ix, 1], uv[ix, 0]]
            valid = pred[side]['valid'][uv[ix, 1], uv[ix, 0]]
            confidence = qdepth[side][uv[ix, 1], uv[ix, 0]]
            conflict = valid & (confidence >= fusion['minimum_depth_confidence_for_free_space'])
            conflict &= z[ix] < measured-fusion['free_space_tolerance_m']
            usable[ix[conflict]] = False
            blocked[side] = int(conflict.sum())
        if qc < fusion['minimum_completion_confidence']:
            usable[:] = False
        fused = np.concatenate([corrected, registered[usable]])
        origins = np.r_[np.ones(len(clean), np.uint8), np.full(usable.sum(), 2, np.uint8)]
        conf = np.r_[qv[eligible], np.full(usable.sum(), qc)]
        ix = voxel_indices(fused, fusion['voxel_m'])
        fused, origins, conf = fused[ix], origins[ix], conf[ix]
    assert np.isfinite(fused).all() and len(fused)
    out.mkdir(parents=True, exist_ok=True)
    fused = fused.astype(np.float32)
    scene = original_scene.copy() if fallback else np.concatenate([original_scene, fused])
    np.savez_compressed(out/'cloud.npz', target_points_m=fused, scene_points_m=scene,
        point_origin=origins, point_confidence=conf.astype(np.float32))
    np.savez_compressed(out/'confidence.npz', left_depth_confidence=qdepth['left'], right_depth_confidence=qdepth['right'],
        left_foreground_confidence=fg['left'], right_foreground_confidence=fg['right'])
    clean_out = output.parent/'confidence_only'/sid
    clean_out.mkdir(parents=True, exist_ok=True)
    filtered = clean if len(clean) >= registration['minimum_observations'] else observed
    np.savez_compressed(clean_out/'cloud.npz', target_points_m=filtered.astype(np.float32), scene_points_m=original_scene)
    record = dict(scene_id=sid, config=config, input_sha256=hashes, code_sha256=sha(__file__),
        seconds=time.perf_counter()-start, observed_points=len(observed), filtered_points=len(clean),
        fused_points=len(fused), original_scene_points_retained=len(original_scene), fallback=bool(fallback),
        registration_confidence=qc, best_registration=best, trials=trials,
        free_space_conflicts=blocked, foreground=foreground_records,
        cloud_sha256=sha(out/'cloud.npz'), ground_truth_input=False)
    write(out/'record.json', record)
    print('FUSED', sid, len(observed), len(clean), len(fused), 'fallback', fallback, 'q_c', round(qc, 3), flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--reverse', type=Path, default=ROOT/'results/development/reverse_stereo')
    p.add_argument('--config', type=Path, default=ROOT/'protocols/adaptive_v1.json')
    p.add_argument('--output', type=Path, default=ROOT/'results/development/adaptive_v1/perception')
    p.add_argument('--scene', action='append')
    a = p.parse_args()
    forbidden = [str((a.source/'datasets'/name).resolve()) for name in ['evaluation', 'scene_specs']]
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(Path(args[0]).resolve())
            if any(path == x or path.startswith(x+'/') for x in forbidden):
                raise PermissionError('Fusion cannot open ground-truth evaluation inputs')
    sys.addaudithook(guard)
    config = read(a.config)
    scenes = a.scene or [s['scene_id'] for s in read(a.source/'datasets/manifest.json')['scenes']]
    for sid in scenes:
        process(a.source, a.reverse, a.output, config, sid)
    write(a.output.parent/'fusion_status.json', dict(status='completed', scenes=len(scenes), config_sha256=sha(a.config), code_sha256=sha(__file__)))


if __name__ == '__main__':
    main()
