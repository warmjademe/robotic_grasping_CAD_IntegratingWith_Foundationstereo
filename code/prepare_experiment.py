"""Prepare a new seed-zero experiment without old run directories or queues."""
import argparse
import json
from pathlib import Path
import shutil
from project_common import read, sha, write

RELEASE = Path(__file__).resolve().parents[1]
STEREO = ('stereobm', 'sgm', 'psmnet', 'gwcnet', 'raft_stereo', 'igev_stereo', 'foundationstereo')
OTHER_GRASPS = ('6dof_graspnet', 'contact_graspnet', 'gpd', 'graspnet_baseline')
METHODS = ([dict(arm='no_cad', perception='foundationstereo', grasp='graspgen'),
            dict(arm='cad_fused', perception='cad_fused', grasp='graspgen')]
           + [dict(arm=p+'__graspgen', perception=p, grasp='graspgen') for p in STEREO if p!='foundationstereo']
           + [dict(arm='foundationstereo__'+g, perception='foundationstereo', grasp=g) for g in OTHER_GRASPS])


def prepare(workspace, menagerie, batch_index, source_ablations):
    scenes = read(RELEASE/'data/scene_index.json')
    # Preserve the original 29 batches rather than inventing a new grouping.
    mapping = read(RELEASE/'configs/batch_objects.json')
    selected = [s for s in scenes if s['object_id'] in mapping[batch_index]]
    study = workspace/('rq3_source_ablation_seed0_v1' if source_ablations else 'comparison_seed0_v1')
    run = study/'batches'/f'batch_{batch_index:03d}'
    assert not run.exists(), 'Preserve existing preparation: '+str(run)
    manifest=dict(scenes=scenes,scene_count=len(scenes),objects=58)
    manifest_path=study/'datasets/manifest.json'
    if manifest_path.exists():assert read(manifest_path)==manifest
    else:write(manifest_path,manifest)
    methods = METHODS
    if source_ablations:
        methods = [dict(arm=p+'__graspgen', perception=p, grasp='graspgen') for p in ('registered_cad','direct_concat')]
        main=workspace/'comparison_seed0_v1/results/outcomes.json'
        original=read(main)
        assert original['status']=='complete' and len(original['rows'])==16704, 'Complete main comparison and aggregate first'
        reused=[row for row in original['rows'] if row['method'] in ('no_cad','cad_fused')]
        assert len(reused)==2784
        reuse_path=study/'results/reused_outcomes.json'
        reuse_record=dict(status='audited_reuse',rows=reused,source_sha256=sha(main))
        if reuse_path.exists():assert read(reuse_path)==reuse_record
        else:write(reuse_path,reuse_record)
        queue=dict(runs=[dict(path=str((study/'batches'/f'batch_{i:03d}').relative_to(workspace))) for i in range(29)])
        if (study/'queue.json').exists():assert read(study/'queue.json')==queue
        else:write(study/'queue.json',queue)
    for folder in ('protocols','datasets/assets','datasets/inputs','datasets/scene_specs',
                   'datasets/evaluation','inputs','assets','ops','logs','results/grasps','results/perception'):
        (run/folder).mkdir(parents=True)
    shutil.copytree(RELEASE/'code/runtime', run/'code')
    # The exporter uses a small path/JSON helper; copy it into the run snapshot.
    shutil.copy2(RELEASE/'code/project_common.py', run/'code/project_common.py')
    catalog = {v['object_id']:v for v in read(RELEASE/'data/canonical_catalog.json')}
    required = set()
    for scene in selected:
        sid = scene['scene_id']; src = RELEASE/'data/scenes'/sid
        spec = read(src/'scene_spec.json')
        write(run/'datasets/scene_specs'/(sid+'.json'), spec)
        write(run/'datasets/inputs'/sid/'task.json', read(src/'task.json'))
        required.update(i['object_id'] for i in spec['instances'])
    asset_catalog = []
    for oid in sorted(required):
        source = workspace/catalog[oid]['mesh']
        assert source.is_file() and sha(source)==catalog[oid]['mesh_sha256'], oid
        (run/'datasets/assets'/oid).symlink_to(source.parent.resolve(), target_is_directory=True)
        asset_catalog.append(dict(catalog[oid], mesh=f'datasets/assets/{oid}/mesh_m.ply'))
    write(run/'datasets/object_catalog.json', asset_catalog)
    write(run/'datasets/manifest.json', dict(scenes=selected, scene_count=len(selected),
        dataset='public_cad_seed0', image_source='New calibrated simulation images; original-size public CAD'))
    tasks = [dict(task_id=f"{s['scene_id']}__{m['arm']}__seed0", episode_id=s['scene_id'],
                  object_id=s['object_id'], dataset=s['dataset'], condition=s['condition'],
                  split='comparison_seed0', seed=0, rqs=[], **m) for s in selected for m in methods]
    write(run/'inputs/manifest.json', dict(tasks=tasks, expected=len(tasks)))
    write(run/'protocols/task_identity.json', tasks)
    protocol = read(RELEASE/'configs/protocol.json')
    protocol['dataset']['scene_count'] = len(selected)
    protocol['comparison'].update(source_batch=f'batch_{batch_index:03d}', scene_count=len(selected),
        grasp_generation_tasks=len(tasks), outcome_rows=2*len(tasks))
    protocol['release_reproduction'] = 'New execution; original published result rows are not overwritten or relabeled'
    if source_ablations:
        protocol['stereo_methods']=['registered_cad','direct_concat']
        protocol['grasp_methods']=['graspgen']
        protocol['unique_combinations']=2
        source=workspace/'comparison_seed0_v1/batches'/run.name
        assert (source/'results/physical_trace_audit.json').exists(), 'Complete and audit source comparison first'
        for name in ('inputs','evaluation'):
            shutil.rmtree(run/'datasets'/name)
            (run/'datasets'/name).symlink_to(source/'datasets'/name,target_is_directory=True)
        for method in ('foundationstereo','cad_fused'):
            (run/'results/perception'/method).symlink_to(source/'results/perception'/method,target_is_directory=True)
        write(run/'ops/imports.json', dict(source=str(source), shared_from=str(source)))
    write(run/'protocols/protocol.json', protocol)
    shutil.copy2(RELEASE/'configs/adaptive_v6.json', run/'protocols/adaptive_v6.json')
    (run/'assets/menagerie').symlink_to(menagerie.resolve(), target_is_directory=True)
    shutil.copy2(RELEASE/'dependencies/assets/calibration_cube.obj', run/'assets/calibration_cube.obj')
    write(run/'ops/prepared.json', dict(status='prepared', release_reproduction=True,
        protocol_sha256=sha(run/'protocols/protocol.json'),
        code_sha256={p.name:sha(p) for p in (run/'code').glob('*.py')}))
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--menagerie',type=Path,required=True)
    parser.add_argument('--batch',type=int,required=True,choices=range(29))
    parser.add_argument('--source-ablations',action='store_true')
    args=parser.parse_args()
    print(prepare(args.workspace.resolve(),args.menagerie,args.batch,args.source_ablations))
