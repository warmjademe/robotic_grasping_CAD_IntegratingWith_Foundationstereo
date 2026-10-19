"""Build fixed whole-round task identities from the published final selection."""
from collections import defaultdict
import json
from pathlib import Path

STEREO = ('stereobm', 'sgm', 'psmnet', 'gwcnet', 'raft_stereo',
          'igev_stereo', 'foundationstereo')
GRASPS = ('graspgen', '6dof_graspnet', 'contact_graspnet', 'gpd', 'graspnet_baseline')
PLATFORMS = ('franka_isaac', 'franka_mujoco')
PROFILES = ('rq1', 'rq2_perception', 'rq2_cross_grasp', 'rq3')


def method(perception, grasp='graspgen'):
    arm = ('no_cad' if perception == 'foundationstereo' else 'cad_fused') if (
        grasp == 'graspgen' and perception in ('foundationstereo', 'cad_fused')) else perception + '__' + grasp
    return dict(arm=arm, perception=perception, grasp=grasp)


def methods(profile):
    if profile in ('rq1', 'rq2_perception'):
        return [method(p) for p in (*STEREO, 'cad_fused')]
    if profile == 'rq2_cross_grasp':
        return [method(p, g) for p in ('foundationstereo', 'cad_fused') for g in GRASPS]
    if profile == 'rq3':
        return [method(p) for p in ('foundationstereo', 'registered_cad', 'direct_concat', 'cad_fused')]
    raise ValueError('Unknown profile: ' + profile)


def selection_index(selection):
    assert selection['status'] == 'final_whole_round_selection'
    assert selection['per_object_seed_mixing'] is False
    configuration = json.loads((Path(__file__).resolve().parents[1] /
        'configs/execution_seeds.json').read_text())
    assert configuration['status'] == 'complete_final_execution_configuration'
    registered = {(r['dataset'], r['platform'], r['grasp']): r for r in configuration['cad_conditions']}
    universe = {(d, p, g) for d in ('tless', 'itodd') for p in PLATFORMS for g in GRASPS}
    assert len(registered) == len(configuration['cad_conditions']) == 20 and set(registered) == universe
    index = {key: r['seed'] for key, r in registered.items() if r['source'] == 'unchanged_original'}
    assert len(index) == 10 and all(seed == 0 for seed in index.values())
    for row in selection['selected_conditions']:
        key = row['dataset'], row['platform'], row['grasp']
        assert key not in index and row['platform'] in PLATFORMS and row['grasp'] in GRASPS
        assert type(row['seed']) is int and row['seed'] in row['eligible_seeds']
        assert registered[key]['source'] == 'selected' and registered[key]['seed'] == row['seed']
        index[key] = row['seed']
    assert set(index) == universe, 'Missing final selected condition'
    return index


def make_tasks(scenes, profile, selection, platforms=PLATFORMS):
    """Share candidates only when scene, perception, grasp and seed agree."""
    assert platforms and len(set(platforms)) == len(platforms)
    assert set(platforms) <= set(PLATFORMS)
    selected = selection_index(selection)
    tasks = []
    for scene in scenes:
        for component in methods(profile):
            by_seed = defaultdict(list)
            if profile == 'rq1':
                by_seed[0] = []
            else:
                for platform in platforms:
                    key = scene['dataset'], platform, component['grasp']
                    seed = selected[key] if component['perception'] == 'cad_fused' else 0
                    by_seed[seed].append(platform)
            for seed, targets in sorted(by_seed.items()):
                tasks.append(dict(
                    task_id=f"{scene['scene_id']}__{component['arm']}__seed{seed}",
                    episode_id=scene['scene_id'], object_id=scene['object_id'],
                    dataset=scene['dataset'], condition=scene['condition'],
                    split=profile, seed=seed, platforms=targets,
                    role='geometry_only' if profile == 'rq1' else 'grasp_execution',
                    rqs=['RQ1' if profile == 'rq1' else 'RQ3' if profile == 'rq3' else 'RQ2'],
                    **component))
    assert len({t['task_id'] for t in tasks}) == len(tasks)
    identities = [(t['episode_id'], t['arm'], p) for t in tasks for p in t['platforms']]
    assert len(set(identities)) == len(identities)
    return tasks
