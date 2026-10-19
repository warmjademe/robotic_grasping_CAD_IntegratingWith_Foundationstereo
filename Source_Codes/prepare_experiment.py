"""Prepare an isolated reproduction batch from the paper's final configuration."""
import argparse
from pathlib import Path
import shutil
from project_common import read, sha, write
from RQ2.experiment_design import PLATFORMS, PROFILES, make_tasks, methods

RELEASE = Path(__file__).resolve().parent
SELECTION = RELEASE / 'RQ2/data/paper_final_20261019/selection.json'


def prepare(workspace, menagerie, batch_index, profile, platforms=PLATFORMS):
    workspace, menagerie = Path(workspace).resolve(), Path(menagerie).resolve()
    scenes = read(RELEASE / 'datasets/scene_index.json')
    mapping = read(RELEASE / 'configs/batch_objects.json')
    selected = [s for s in scenes if s['object_id'] in mapping[batch_index]]
    selection = read(SELECTION)
    assert read(RELEASE / 'configs/execution_seeds.json')['selection_sha256'] == sha(SELECTION)
    tasks = make_tasks(selected, profile, selection, platforms)
    study = workspace / (profile + '_paper_final')
    run = study / 'batches' / f'batch_{batch_index:03d}'
    assert not run.exists(), 'Preserve existing preparation: ' + str(run)
    manifest = dict(scenes=scenes, scene_count=len(scenes), objects=58)
    manifest_path = study / 'datasets/manifest.json'
    if manifest_path.exists():
        assert read(manifest_path) == manifest
    else:
        write(manifest_path, manifest)
    for folder in ('protocols', 'datasets/assets', 'datasets/inputs', 'datasets/scene_specs',
                   'datasets/evaluation', 'inputs', 'assets', 'ops', 'logs', 'results/grasps',
                   'results/perception'):
        (run / folder).mkdir(parents=True)
    shutil.copytree(RELEASE / 'runtime', run / 'code')
    shutil.copy2(RELEASE / 'project_common.py', run / 'code/project_common.py')
    catalog = {v['object_id']: v for v in read(RELEASE / 'datasets/canonical_catalog.json')}
    required = set()
    for scene in selected:
        sid = scene['scene_id']
        src = RELEASE / 'datasets/scenes' / sid
        spec = read(src / 'scene_spec.json')
        write(run / 'datasets/scene_specs' / (sid + '.json'), spec)
        write(run / 'datasets/inputs' / sid / 'task.json', read(src / 'task.json'))
        required.update(i['object_id'] for i in spec['instances'])
    asset_catalog = []
    for oid in sorted(required):
        source = workspace / catalog[oid]['mesh']
        assert source.is_file() and sha(source) == catalog[oid]['mesh_sha256'], oid
        samples = source.with_name('cad_samples.npz')
        assert samples.is_file() and sha(samples) == catalog[oid]['samples_sha256'], oid
        (run / 'datasets/assets' / oid).symlink_to(source.parent, target_is_directory=True)
        asset_catalog.append(dict(catalog[oid], mesh=f'datasets/assets/{oid}/mesh_m.ply'))
    write(run / 'datasets/object_catalog.json', asset_catalog)
    write(run / 'datasets/manifest.json', dict(scenes=selected, scene_count=len(selected),
        dataset='public_cad_final_reproduction',
        image_source='New calibrated simulation images; original-size public CAD'))
    write(run / 'inputs/manifest.json', dict(tasks=tasks, expected=len(tasks)))
    write(run / 'protocols/task_identity.json', tasks)
    protocol = read(RELEASE / 'configs/protocol.json')
    protocol['dataset']['scene_count'] = len(selected)
    protocol['platforms'] = [] if profile == 'rq1' else list(platforms)
    protocol['release_profile'] = profile
    protocol['purpose'] = profile + ': reproduce the published per-condition whole round'
    protocol['grasp_seeds'] = sorted({t['seed'] for t in tasks}) if profile != 'rq1' else []
    protocol['stereo_methods'] = list(dict.fromkeys(t['perception'] for t in tasks))
    protocol['grasp_methods'] = [] if profile == 'rq1' else list(dict.fromkeys(t['grasp'] for t in tasks))
    protocol['unique_combinations'] = len(methods(profile))
    protocol['comparison'].update(source_batch=f'batch_{batch_index:03d}',
        scene_count=len(selected), grasp_generation_tasks=0 if profile == 'rq1' else len(tasks),
        outcome_rows=sum(len(t['platforms']) for t in tasks),
        methods=[m['arm'] for m in methods(profile)],
        seeds=sorted({t['seed'] for t in tasks}), selection_sha256=sha(SELECTION))
    protocol['release_reproduction'] = (
        'New execution of published whole-round seeds; original result rows remain unchanged. '
        'Selected comparisons are descriptive best-observed results, not a new selection run.')
    write(run / 'protocols/protocol.json', protocol)
    shutil.copy2(RELEASE / 'configs/adaptive_v6.json', run / 'protocols/adaptive_v6.json')
    shutil.copy2(SELECTION, run / 'protocols/paper_selection.json')
    shutil.copy2(RELEASE / 'configs/execution_seeds.json', run / 'protocols/execution_seeds.json')
    if profile == 'rq3':
        write(run / 'ops/imports.json', dict(source=str(run)))
    (run / 'assets/menagerie').symlink_to(menagerie, target_is_directory=True)
    shutil.copy2(RELEASE / 'dependencies/assets/calibration_cube.obj', run / 'assets/calibration_cube.obj')
    write(run / 'ops/prepared.json', dict(status='prepared', release_reproduction=True,
        protocol_sha256=sha(run / 'protocols/protocol.json'), selection_sha256=sha(SELECTION),
        task_identity_sha256=sha(run / 'protocols/task_identity.json'),
        execution_seeds_sha256=sha(run / 'protocols/execution_seeds.json'),
        code_sha256={p.name: sha(p) for p in (run / 'code').glob('*.py')}))
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--menagerie', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True, choices=range(29))
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--platform', choices=PLATFORMS, action='append')
    args = parser.parse_args()
    print(prepare(args.workspace, args.menagerie, args.batch, args.profile, args.platform or PLATFORMS))
