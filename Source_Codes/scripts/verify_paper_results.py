"""Verify the final manuscript data locally using Python's standard library.

This checks release bytes and reproduces statistics from released rows. Original
physics/trajectory audits are identified by hash; they are not rerun here.
"""
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

FINAL = 'paper_final_20261019'
ARMS = {'no_cad': 'A1', 'registered_cad__graspgen': 'A2', 'direct_concat__graspgen': 'A3', 'cad_fused': 'A4'}
PINNED_SELECTION = '069cb0d7f86e71b6b40d9046501abe8a81b78529994fe59aaec73842ec43d7ff'
PRIVATE = re.compile(r'/(?:Users|home|root)/|192\.168\.|(?:sk-|ghp_|olp_|hf_)[A-Za-z0-9_-]{20,}')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row['scene_id'], row['platform'], row['method']


def close(actual, expected, label):
    need(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10), 'Numeric mismatch: ' + label)


def outcome_groups(data, scene_index, expected_methods):
    need(len({key(r) for r in data}) == len(data), 'Duplicate outcome identity')
    need({r['method'] for r in data} == set(expected_methods), 'Outcome method set mismatch')
    cells = defaultdict(list)
    for r in data:
        need(type(r['success']) is bool and type(r['executed']) is bool, 'Pending/invalid outcome')
        need(r.get('outcome_status', 'completed') == 'completed', 'Unfinished outcome')
        need(type(r['seed']) is int, 'Invalid seed')
        scene = scene_index[r['scene_id']]
        need((scene['dataset'], scene['object_id']) == (r['dataset'], r['object_id']), 'Outcome scene metadata mismatch')
        if r['executed']:
            need(r['physics_evidence_verified'] and r['evidence'] and r['result_sha256'] and r['trace_sha256'], 'Missing physics provenance')
        else:
            need(not r['success'] and r['planning_status'] != 'sampled_path_found', 'Invalid no-path outcome')
        cells[r['dataset'], r['platform'], r['method']].append(r)
    need(len(cells) == 4 * len(expected_methods), 'Missing dataset/platform/method cell')
    result = {}
    for cell, records in cells.items():
        ds, platform, method = cell
        need(platform in {'franka_isaac', 'franka_mujoco'}, 'Unexpected platform')
        expected_scenes = {sid for sid, scene in scene_index.items() if scene['dataset'] == ds}
        need({r['scene_id'] for r in records} == expected_scenes and len(records) == len(expected_scenes), 'Incomplete scene denominator')
        seeds = {r['seed'] for r in records}
        need(len(seeds) == 1, 'Seeds mixed within a complete condition')
        objects = defaultdict(list)
        for r in records:
            objects[r['object_id']].append(r)
        need(set(map(len, objects.values())) == {12}, 'Object scene count is not twelve')
        covered = sorted(o for o, values in objects.items() if any(r['success'] for r in values))
        executions = sum(r['executed'] for r in records)
        successes = sum(r['success'] for r in records)
        result[cell] = dict(dataset=ds, platform=platform, method=method, seed=next(iter(seeds)),
                            objects=len(objects), covered=len(covered), scenes=len(records),
                            successes=successes, executed=executions, no_path=len(records)-executions,
                            failures=len(records)-successes, physical_failures=executions-successes,
                            object_success_rate=len(covered)/len(objects), scene_success_rate=successes/len(records),
                            successful_object_ids=covered,
                            planning_status_counts=dict(Counter(r['planning_status'] for r in records)))
    return result


def check_groups(actual, reported):
    need(len(actual) == len(reported), 'Group count mismatch')
    for group in reported:
        value = actual[group['dataset'], group['platform'], group['method']]
        for field in value:
            need(value[field] == group[field], 'Outcome group mismatch: ' + str((field, value, group)))


def geometry_groups(data, scene_index, methods, subset=False):
    need(len({(r['scene_id'], r['method']) for r in data}) == len(data), 'Duplicate geometry identity')
    need({r['method'] for r in data} == set(methods), 'Geometry methods mismatch')
    for method in methods:
        need({r['scene_id'] for r in data if r['method'] == method} == set(scene_index), 'Missing full geometry denominator')
    references = defaultdict(set)
    cells = defaultdict(list)
    for r in data:
        scene = scene_index[r['scene_id']]
        need((scene['dataset'], scene['object_id']) == (r['dataset'], r['object_id']), 'Geometry metadata mismatch')
        references[r['scene_id']].add(r['reference_surface_sha256'])
        need(r['cloud_sha256'] and r['evaluation_scene_sha256'], 'Missing geometry source hashes')
        if not subset or r['common_subset_accepted']:
            cells[r['dataset'], r['method']].append(r)
    need(all(len(v) == 1 for v in references.values()), 'Geometry reference differs between methods')
    result = {}
    for (ds, method), records in cells.items():
        objects = defaultdict(list)
        for r in records:
            objects[r['object_id']].append(r)
        value = dict(dataset=ds, method=method, scenes=len(records), objects=len(objects),
                     empty_scenes=sum(r['points'] == 0 for r in records))
        for metric in ['surface_mae_mm', 'precision_2mm', 'complete_coverage_2mm', 'points']:
            object_means = [fmean(r[metric] for r in rs if r[metric] is not None)
                            for rs in objects.values() if any(r[metric] is not None for r in rs)]
            value[metric] = fmean(object_means) if object_means else None
        result[ds, method] = value
    return result


def check_geometry(actual, reported):
    need(len(actual) == len(reported), 'Geometry group count mismatch')
    for g in reported:
        value = actual[g['dataset'], g['method']]
        for field in value:
            if isinstance(value[field], float):
                close(value[field], g[field], 'geometry ' + field)
            else:
                need(value[field] == g[field], 'Geometry count mismatch: ' + field)


def verify(root):
    index = read(root / 'data/paper_results_index.json')
    need(index['original_selection_sha256'] == PINNED_SELECTION, 'Final selection source differs')
    need(index['source_files']['final_selection']['sha256'] == PINNED_SELECTION, 'Selection provenance differs')
    listed = set()
    for file in index['files']:
        rel = file['release_path']
        need(rel not in listed and not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'Invalid/duplicate release path')
        listed.add(rel)
        path = root / rel
        need(path.is_file() and path.stat().st_size == file['bytes'] and digest(path) == file['sha256'], 'Release hash mismatch: ' + rel)
        # The verifier source itself declares the forbidden marker expressions.
        if path.suffix != '.py':
            need(not PRIVATE.search(path.read_text()), 'Private path/credential marker: ' + rel)
        if 'rows' in file:
            need(len(rows(path)) == file['rows'], 'Release row count mismatch: ' + rel)
    need(not PRIVATE.search((root / 'data/paper_results_index.json').read_text()), 'Private index metadata')
    scenes = read(root / 'data/paper_scene_index.json')
    scene_index = {r['scene_id']: r for r in scenes}
    need(len(scene_index) == len(scenes) == 696, 'Scene count mismatch')
    need(Counter(r['dataset'] for r in scenes) == {'tless': 360, 'itodd': 336}, 'Dataset scene count mismatch')
    need(Counter(len(rs) for rs in _object_lists(scenes).values()) == {12: 58}, 'Object denominator mismatch')
    q1 = root / 'RQ1/data' / FINAL
    q2 = root / 'RQ2/data' / FINAL
    q3 = root / 'RQ3/data' / FINAL
    g1, s1 = rows(q1 / 'geometry.jsonl'), read(q1 / 'summary.json')
    baselines = rows(q2 / 'perception_baselines.jsonl')
    comparisons = rows(q2 / 'cad_comparisons.jsonl')
    s2, selection = read(q2 / 'summary.json'), read(q2 / 'selection.json')
    ablation, g3, s3 = rows(q3 / 'outcomes.jsonl'), rows(q3 / 'geometry.jsonl'), read(q3 / 'summary.json')
    baseline_methods = ['no_cad', 'stereobm__graspgen', 'sgm__graspgen', 'psmnet__graspgen', 'gwcnet__graspgen', 'raft_stereo__graspgen', 'igev_stereo__graspgen']
    compare_methods = ['no_cad', 'cad_fused'] + [p + '__' + g for p in ['foundationstereo', 'cad_fused'] for g in ['6dof_graspnet', 'contact_graspnet', 'gpd', 'graspnet_baseline']]
    need((len(g1), len(baselines), len(comparisons), len(ablation), len(g3)) == (5568, 9744, 13920, 5568, 2784), 'Final data row counts mismatch')
    b_groups = outcome_groups(baselines, scene_index, baseline_methods)
    c_groups = outcome_groups(comparisons, scene_index, compare_methods)
    a_groups = outcome_groups(ablation, scene_index, ARMS)
    check_groups(b_groups, s2['perception_baseline_groups'])
    check_groups(c_groups, s2['groups'])
    check_groups(a_groups, s3['groups'])
    need(all(r['seed'] == 0 for r in baselines), 'Original baseline seed changed')
    c_lookup = {key(r): r for r in comparisons}
    for row in ablation + [r for r in baselines if r['method'] == 'no_cad']:
        if row['method'] in ['no_cad', 'cad_fused']:
            shared = c_lookup[key(row)]
            for field in ['success', 'executed', 'seed', 'planning_status', 'plan_sha256', 'evidence', 'result_sha256', 'trace_sha256']:
                need(row.get(field) == shared.get(field), 'Shared RQ2/RQ3 trial mismatch: ' + field)
    object_directions, scene_directions = Counter(), Counter()
    for p in s2['comparisons']:
        ds, platform, grasp = p['dataset'], p['platform'], p['grasp']
        no_cad = 'no_cad' if grasp == 'graspgen' else 'foundationstereo__' + grasp
        cad = 'cad_fused' if grasp == 'graspgen' else 'cad_fused__' + grasp
        b, a = c_groups[ds, platform, no_cad], c_groups[ds, platform, cad]
        for tag, group in [('no_cad', b), ('cad', a)]:
            for field in ['objects', 'covered', 'scenes', 'successes', 'seed', 'executed', 'no_path']:
                need(group[field] == p[tag][field], 'RQ2 paired summary mismatch')
        gained, lost = sorted(set(a['successful_object_ids']) - set(b['successful_object_ids'])), sorted(set(b['successful_object_ids']) - set(a['successful_object_ids']))
        need(gained == p['gained_objects'] and lost == p['lost_objects'], 'Paired object identities mismatch')
        original = [r for r in comparisons if (r['dataset'], r['platform'], r['method']) == (ds, platform, no_cad)]
        scene_gain = sum(c_lookup[r['scene_id'], platform, cad]['success'] and not r['success'] for r in original)
        scene_loss = sum(r['success'] and not c_lookup[r['scene_id'], platform, cad]['success'] for r in original)
        need((scene_gain, scene_loss) == (p['scene_gained'], p['scene_lost']), 'Paired scene identities mismatch')
        close(100 * (a['covered'] - b['covered']) / a['objects'], p['object_difference_pp'], 'object pp')
        close(100 * (a['successes'] - b['successes']) / a['scenes'], p['scene_difference_pp'], 'scene pp')
        object_directions[_direction(a['covered'] - b['covered'])] += 1
        scene_directions[_direction(a['successes'] - b['successes'])] += 1
        need(p['descriptive_only'] and p['inferential_test'] is None, 'Selected comparisons must remain descriptive')
    need(len(s2['comparisons']) == 20 and object_directions == {'improved': 16, 'tied': 2, 'decreased': 2}, 'RQ2 object directions mismatch')
    need(scene_directions == {'improved': 14, 'tied': 1, 'decreased': 5}, 'RQ2 scene directions mismatch')
    need(selection['source_sha256'] == PINNED_SELECTION and not selection['per_object_seed_mixing'], 'Selection integrity mismatch')
    need(selection['new_seed3_best_count'] == len(selection['new_seed3_best_conditions']) == 6, 'New best count mismatch')
    for p in selection['selected_conditions']:
        g = c_groups[p['dataset'], p['platform'], p['method']]
        for field in ['seed', 'covered', 'scenes', 'successes', 'executed', 'no_path']:
            need(g[field] == p[field], 'Selected whole-round mismatch')
    for ds, seed, covered, successes in [('tless', 0, 17, 76), ('itodd', 1, 15, 49)]:
        g = c_groups[ds, 'franka_mujoco', 'cad_fused']
        need((g['seed'], g['covered'], g['successes']) == (seed, covered, successes), 'Historical GraspGen selection mismatch')
    q1g = geometry_groups(g1, scene_index, baseline_methods + ['cad_fused'])
    q3g = geometry_groups(g3, scene_index, ARMS, subset=True)
    fused_clouds = {r['scene_id']: r['cloud_sha256'] for r in g1 if r['method'] == 'cad_fused'}
    cloud_hash_checks = 0
    for r in comparisons:
        if r['method'].startswith('cad_fused') and r.get('cloud_sha256'):
            need(r['cloud_sha256'] == fused_clouds[r['scene_id']], 'Selected CAD cloud differs from frozen geometry')
            cloud_hash_checks += 1
    check_geometry(q1g, s1['groups'])
    check_geometry(q3g, s3['geometry_common_subset'])
    for p in s1['baseline_contrasts']:
        b, f = q1g[p['dataset'], p['baseline_method']], q1g[p['dataset'], 'cad_fused']
        close(b['surface_mae_mm'], p['baseline_mae_mm'], 'baseline MAE')
        close(f['surface_mae_mm'], p['fused_mae_mm'], 'fused MAE')
        close(100 * (b['surface_mae_mm'] - f['surface_mae_mm']) / b['surface_mae_mm'], p['mae_reduction_percent'], 'MAE reduction')
    accepted = defaultdict(set)
    for r in g3:
        if r['common_subset_accepted']:
            accepted[r['dataset'], r['method']].add(r['scene_id'])
    for ds, n, objects in [('tless', 307, 29), ('itodd', 311, 28)]:
        need(all(len(accepted[ds, m]) == n for m in ARMS), 'Common subset scene count mismatch')
        need(all(accepted[ds, m] == accepted[ds, 'no_cad'] for m in ARMS), 'Common subset differs across arms')
        need(all(q3g[ds, m]['objects'] == objects for m in ARMS), 'Common subset object count mismatch')
    unique = {key(r) for r in baselines + comparisons + ablation}
    need(len(unique) == 25056, 'Cross-RQ deduplicated outcome count mismatch')
    return dict(status='passed', release_files_verified=len(listed), scenes=696, objects=58,
                rq1_geometry_rows=len(g1), rq2_perception_baseline_rows=len(baselines),
                rq2_cad_comparison_rows=len(comparisons), rq3_outcome_rows=len(ablation),
                rq3_geometry_rows=len(g3), unique_released_outcome_trials=len(unique),
                rq2_object_directions=dict(object_directions), rq2_scene_directions=dict(scene_directions),
                new_seed3_best_conditions=6, inferred_new_physics_executions=0,
                selected_cad_cloud_hashes_compared=cloud_hash_checks,
                original_selection_sha256=PINNED_SELECTION,
                verification_scope='Release SHA-256, complete row identities, final whole-round selections and offline metric reproduction; original trajectories not replayed')


def _object_lists(scenes):
    groups = defaultdict(list)
    for scene in scenes:
        groups[scene['dataset'], scene['object_id']].append(scene)
    return groups


def _direction(value):
    return 'improved' if value > 0 else 'decreased' if value < 0 else 'tied'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Source_Codes directory')
    args = p.parse_args()
    print(json.dumps(verify(args.root.resolve()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
