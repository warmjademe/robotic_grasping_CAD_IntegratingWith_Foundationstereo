"""Aggregate a complete new reproduction using its registered seeds and denominator."""
import argparse
from collections import defaultdict
from pathlib import Path
from project_common import read, sha, write
from RQ2.experiment_design import PLATFORMS, PROFILES, make_tasks, methods

RELEASE = Path(__file__).resolve().parent


def aggregate(workspace, profile, platforms=PLATFORMS):
    study = Path(workspace) / (profile + '_paper_final')
    scenes = read(RELEASE / 'datasets/scene_index.json')
    selection_path = RELEASE / 'RQ2/data/paper_final_20261019/selection.json'
    tasks = make_tasks(scenes, profile, read(selection_path), platforms)
    expected_geometry = {(t['episode_id'], t['perception']) for t in tasks}
    expected_outcomes = {(t['episode_id'], t['arm'], p): t['seed'] for t in tasks for p in t['platforms']}
    geometry, outcomes, sources = {}, {}, {}
    for index in range(29):
        run = study / 'batches' / f'batch_{index:03d}'
        prepared = read(run / 'ops/prepared.json')
        assert prepared['selection_sha256'] == sha(selection_path)
        protocol = read(run / 'protocols/protocol.json')
        assert protocol['release_profile'] == profile
        assert protocol['platforms'] == ([] if profile == 'rq1' else list(platforms))
        path = run / 'results/geometry.json'
        record = read(path)
        assert record['pending'] == [], 'Pending geometry: ' + str(run)
        for row in record['rows']:
            key = row['scene_id'], row['perception']
            assert key in expected_geometry and key not in geometry
            geometry[key] = row
        sources[str(path.relative_to(study))] = sha(path)
        if profile != 'rq1':
            path = run / 'results/outcomes.json'
            record = read(path)
            assert record['status'] == 'audited' and len(record['rows']) == record['expected_rows']
            assert record['physical_trace_audit_sha256'] == sha(run / 'results/physical_trace_audit.json')
            assert record['manifest_sha256'] == sha(run / 'inputs/manifest.json')
            for row in record['rows']:
                key = row['scene_id'], row['method'], row['platform']
                assert key in expected_outcomes and key not in outcomes
                assert row['seed'] == expected_outcomes[key] and type(row['success']) is bool
                if row['executed']:
                    assert row['physics_evidence_verified'] is True
                else:
                    assert row['success'] is False
                outcomes[key] = row
            sources[str(path.relative_to(study))] = sha(path)
    assert set(geometry) == expected_geometry
    assert set(outcomes) == set(expected_outcomes)
    groups = defaultdict(list)
    for row in outcomes.values():
        groups[row['dataset'], row['method'], row['platform']].append(row)
    grasp_groups = []
    for (dataset, method, platform), rows in sorted(groups.items()):
        objects = {r['object_id'] for r in rows}
        successful_objects = {r['object_id'] for r in rows if r['success']}
        successes = sum(r['success'] for r in rows)
        grasp_groups.append(dict(dataset=dataset, method=method, platform=platform,
            seed=next(iter({r['seed'] for r in rows})), objects=len(objects),
            successful_objects=len(successful_objects), object_success_rate=len(successful_objects)/len(objects),
            attempts=len(rows), successes=successes, attempt_success_rate=successes/len(rows),
            executed=sum(r['executed'] for r in rows), no_path=sum(not r['executed'] for r in rows)))
        assert len({r['seed'] for r in rows}) == 1
    point_groups = defaultdict(list)
    for row in geometry.values():
        point_groups[row['dataset'], row['perception']].append(row)
    geometry_groups = []
    for (dataset, perception), rows in sorted(point_groups.items()):
        item = dict(dataset=dataset, perception=perception, scenes=len(rows))
        for metric in ('surface_mae_mm', 'surface_p95_mm', 'precision_2mm', 'complete_coverage_2mm'):
            values = [r[metric] for r in rows if r[metric] is not None]
            item[metric] = dict(valid=len(values), mean=sum(values)/len(values) if values else None)
        geometry_groups.append(item)
    result = dict(status='complete_audited_reproduction', profile=profile,
        selection_sha256=sha(selection_path), geometry_rows=len(geometry), outcome_rows=len(outcomes),
        grasp_groups=grasp_groups, geometry_groups=geometry_groups, source_sha256=sources,
        scope='New execution of published selected seeds; original published outcomes are separate data.')
    write(study / 'results/summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--platform', choices=PLATFORMS, action='append')
    args = parser.parse_args()
    result = aggregate(args.workspace.resolve(), args.profile, args.platform or PLATFORMS)
    print(result['status'], result['geometry_rows'], result['outcome_rows'])
