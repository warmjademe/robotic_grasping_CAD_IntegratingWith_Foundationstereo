"""Source ablations with frozen estimated poses; never reads evaluation truth."""
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
from project_common import read, write, sha

PERCEPTIONS = ('registered_cad', 'direct_concat')


def construct(observed, cad, transform, minimum_height, voxel_size, accepted):
    registered = cad @ transform[:3, :3].T + transform[:3, 3]
    registered = registered[registered[:, 2] > minimum_height]
    output = {}
    for method in PERCEPTIONS:
        points = registered if method == 'registered_cad' else np.concatenate([observed, registered])
        if not accepted:
            points = np.empty((0, 3), dtype=np.float32)
        if len(points):
            _, indices = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0, return_index=True)
            points = points[np.sort(indices)]
        output[method] = np.asarray(points, dtype=np.float32)
    return output, len(registered)


def build(run):
    sys.path.insert(0, str(run / 'code'))
    from adaptive_fusion_v6 import fallback_reasons
    binding = read(run / 'ops/imports.json')
    for scene in read(run / 'datasets/manifest.json')['scenes']:
        start = time.perf_counter()
        sid, oid = scene['scene_id'], scene['object_id']
        refdir = run / 'results/perception/cad_fused' / sid
        visualdir = run / 'results/perception/foundationstereo' / sid
        ref = read(refdir / 'record.json')
        assert not ref['ground_truth_input'] and sha(refdir / 'cloud.npz') == ref['cloud_sha256']
        # Verify all inputs bound to the original estimation, including CAD samples.
        for path, digest in ref['input_sha256'].items():
            assert sha(path) == digest, path
        assert ref['code_sha256'] == sha(Path(binding['source']) / 'code/adaptive_fusion_v6.py')
        cadpath = run / 'datasets/assets' / oid / 'cad_samples.npz'
        assert str(cadpath.resolve()) in {str(Path(p).resolve()) for p in ref['input_sha256']}, 'CAD sampling provenance missing'
        cfg = ref['config']
        assert cfg == read(run / 'protocols/adaptive_v6.json')
        best = ref['best_registration']
        reasons = fallback_reasons(best, ref['registration_confidence'], ref['surface_support'],
                                   ref['free_space_conflict_fraction'], cfg['registration'], cfg['fusion'])
        assert reasons == ref['fallback_reasons']
        accepted = best is not None and not reasons
        assert bool(ref['fallback']) == (not accepted), 'Frozen gate differs'
        transform = np.asarray(best['estimated_world_from_cad'] if best else np.eye(4), dtype=float)
        assert transform.shape == (4, 4) and np.isfinite(transform).all()
        assert np.allclose(transform[3], [0, 0, 0, 1])
        with np.load(visualdir / 'cloud.npz', allow_pickle=False) as data:
            observed, environment = data['target_points_m'], data['scene_points_m']
        with np.load(cadpath, allow_pickle=False) as data:
            cad = data['points_m']
        assert all(np.isfinite(a).all() for a in (observed, environment, cad))
        output, registered_count = construct(observed, cad, transform,
            cfg['fusion']['cad_surface_minimum_height_m'], cfg['fusion']['voxel_m'], accepted)
        for method, points in output.items():
            dest = run / 'results/perception' / method / sid
            assert not dest.exists(), 'Do not overwrite a prior point-cloud attempt'
            dest.mkdir(parents=True)
            scene_points = environment.copy() if not accepted else np.concatenate([environment, points])
            np.savez_compressed(dest / 'cloud.npz', target_points_m=points, scene_points_m=scene_points)
            os.link(dest / 'cloud.npz', dest / 'grasp_input.npz')
            record = dict(scene_id=sid, object_id=oid, method=method, status='completed',
                accepted=accepted, gate_failure_reasons=reasons, fallback=False,
                original_fusion_fallback=ref['fallback'], estimated_world_from_cad=transform.tolist() if best else None,
                transform_source='frozen stereo-estimated transform; not simulator truth',
                observed_target_points=len(observed), sampled_cad_points=len(cad),
                registered_supported_points=registered_count, target_points=len(points),
                original_scene_points=len(environment), scene_points=len(scene_points),
                voxel_m=cfg['fusion']['voxel_m'], minimum_height_m=cfg['fusion']['cad_surface_minimum_height_m'],
                seconds=time.perf_counter()-start, source_sha256=sha(__file__),
                cad_samples_sha256=sha(cadpath), visual_cloud_sha256=sha(visualdir / 'cloud.npz'),
                frozen_fusion_record_sha256=sha(refdir / 'record.json'), cloud_sha256=sha(dest / 'cloud.npz'),
                geometry_truth_used=False, object_pose_used=False, cad_input=True,
                grasp_input_ready=accepted and min(len(points), len(scene_points)) > 128,
                graspgen_reads='target_points_m only',
                planning_scene_policy='all original observed scene points plus this arm target points')
            if not accepted:
                record.update(upstream_failure_status='upstream_algorithm_failure', failure_stage='registration_gate')
                write(dest / 'failure.json', dict(status='upstream_algorithm_failure', reasons=reasons,
                    registration_record_sha256=record['frozen_fusion_record_sha256'], no_stereo_fallback=True))
                record['upstream_failure_record_sha256'] = sha(dest / 'failure.json')
            write(dest / 'perception.json', record)
            print('SOURCE_CLOUD', sid, method, len(points), 'accepted', accepted, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    build(args.run.resolve())
