"""Add the shared grasp contract to frozen v6 clouds without changing points."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import ROOT, read, sha, write


def arrays_and_metadata(fused, visual):
    import numpy as np
    with np.load(fused / 'cloud.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    with np.load(visual / 'grasp_input.npz', allow_pickle=False) as data:
        raw = {k: data[k] for k in data.files}
    metadata = read(visual / 'grasp_input.json')
    fusion_record = read(fused / 'record.json')
    fallback = fusion_record['fallback']
    assert type(fallback) is bool
    assert metadata['npz_sha256'] == sha(visual / 'grasp_input.npz')
    assert read(fused / 'perception.json')['cloud_sha256'] == sha(fused / 'cloud.npz')
    # v6 appends fused target points to the unmodified observed scene; a
    # rejected fusion instead retains the original scene. Fail closed if this
    # layout changes rather than inventing a point-to-camera association.
    observed = len(raw['scene_points_m'])
    assert np.array_equal(arrays['scene_points_m'][:observed], raw['scene_points_m'])
    appended = len(arrays['scene_points_m']) - observed
    assert appended >= 0
    if appended:
        assert np.array_equal(arrays['scene_points_m'][observed:], arrays['target_points_m'])
    arrays['camera_origins_m'] = raw['camera_origins_m']
    arrays['T_world_reference'] = raw['T_world_reference']
    arrays['point_camera_ids'] = np.concatenate([
        raw['point_camera_ids'], np.full(appended, -1, dtype=raw['point_camera_ids'].dtype)])
    # No colors/normals are manufactured for CAD points. Backends use their
    # existing geometry-only interface and unchanged visibility adaptation.
    metadata = dict(metadata, cad_used=not fallback, frozen_fusion_fallback=fallback,
        frozen_fusion_fallback_reasons=fusion_record['fallback_reasons'],
        perception_route='cad_stereo_fusion_v6',
        has_colors='scene_colors' in arrays, has_normals='normals_world' in arrays,
        target_selection='frozen_v6_fusion_with_original_fallback',
        input_hashes=dict(metadata['input_hashes'],
            frozen_fused_cloud=sha(fused / 'cloud.npz'),
            frozen_fusion_record=sha(fused / 'record.json'),
            source_stereo_grasp_input=sha(visual / 'grasp_input.npz')),
        camera_provenance='Original observed scene IDs retained; appended fusion points have ID -1',
        additional_point_provenance='CAD or fused estimates; camera visibility is not a measurement',
        point_coordinates_changed=False, adapter_source_sha256=sha(__file__))
    metadata.pop('npz_sha256', None)
    return arrays, metadata


def contract_module():
    file = Path(os.environ['GRASP_MODULE_ROOT']) / 'shared/contract.py'
    spec = importlib.util.spec_from_file_location('rq2_frozen_grasp_contract', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_inputs(run, validate_only=False):
    import numpy as np
    contract = contract_module()
    records = []
    source = run / 'results/perception'
    assert not (run / 'results/input_contract_audit.json').exists(), 'Preserve prior adapter evidence'
    for scene in read(run / 'datasets/manifest.json')['scenes']:
        sid = scene['scene_id']
        pair = dict(cad_fused=source / 'cad_fused' / sid,
                    foundationstereo=source / 'foundationstereo' / sid)
        ready = read(pair['cad_fused'] / 'perception.json')['grasp_input_ready']
        if not ready and not (pair['foundationstereo'] / 'grasp_input.npz').exists():
            records.append(dict(scene_id=sid, ready=False,
                status='upstream_insufficient_points', coordinate_identity_verified=True,
                source_cloud_sha256=sha(pair['cad_fused'] / 'cloud.npz')))
            continue
        arrays, metadata = arrays_and_metadata(pair['cad_fused'], pair['foundationstereo'])
        if ready:
            contract.validate(arrays, metadata)
        folder = run / 'results/perception/cad_fused' / sid
        input_path = folder / 'grasp_input.npz'
        record = dict(scene_id=sid, ready=ready, target_points=len(arrays['target_points_m']),
            frozen_fusion_fallback=metadata['frozen_fusion_fallback'], cad_used=metadata['cad_used'],
            scene_points=len(arrays['scene_points_m']),
            unknown_camera_points=int((arrays['point_camera_ids'] < 0).sum()),
            source_cloud_sha256=sha(pair['cad_fused'] / 'cloud.npz'),
            source_visual_input_sha256=sha(pair['foundationstereo'] / 'grasp_input.npz'),
            coordinate_identity_verified=True)
        if not validate_only:
            # The export stage produces an unadapted copy of cloud.npz. Only
            # replace this exact placeholder before candidate generation.
            assert input_path.exists() and sha(input_path) == record['source_cloud_sha256'], input_path
            assert not input_path.with_suffix('.json').exists(), input_path
            assert not any((run / 'results/grasps').glob(sid + '__*/task_result.json')), 'Candidates already exist'
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(input_path, **arrays)
            write(input_path.with_suffix('.json'), dict(metadata, npz_sha256=sha(input_path)))
            with np.load(input_path) as output, np.load(pair['cad_fused'] / 'cloud.npz') as original:
                assert all(output[k].dtype == original[k].dtype and
                           np.array_equal(output[k], original[k]) for k in original.files)
            if ready:
                contract.load(input_path)
            record['adapted_input_sha256'] = sha(input_path)
        records.append(record)
    if not validate_only:
        write(run / 'results/input_contract_audit.json', dict(status='passed', rows=records,
            contract_sha256=sha(contract.__file__), source_sha256=sha(__file__),
            statement='Every original frozen fused array is preserved exactly; only camera metadata is added'))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    records = prepare_inputs(args.run, args.validate_only)
    print('RQ2_INPUT_CONTRACT_CHECKED', len(records), flush=True)
