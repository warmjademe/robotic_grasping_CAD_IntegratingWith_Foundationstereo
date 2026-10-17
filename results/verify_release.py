#!/usr/bin/env python3
"""Verify released outcome denominators, metrics, provenance and shared arms.

Run from any directory with Python 3.8+. This reads release files only. It does
not execute robot physics, download data, or claim an independent trajectory
replay. Output is a JSON verification receipt on stdout.
"""
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def verify_study(name, scenes):
    folder = ROOT / 'results' / name
    summary = load(folder / 'summary.json')
    rows = [json.loads(line) for path in sorted((folder / 'outcomes').glob('*.jsonl'))
            for line in path.read_text().splitlines() if line.strip()]
    keys = [(r['scene_id'], r['method'], r['platform'], r['seed']) for r in rows]
    expected = set(itertools.product(scenes, summary['methods'], summary['platforms'], [0]))
    assert len(keys) == len(set(keys)), 'Duplicate outcomes'
    assert set(keys) == expected, 'Missing or unexpected outcomes'
    assert len(rows) == summary['expected_outcomes'] == summary['completed_outcomes']
    assert summary['pending_outcomes'] == 0
    for row in rows:
        assert row['outcome_status'] == 'completed'
        assert type(row['success']) is bool and type(row['executed']) is bool
        assert row['object_id'] == scenes[row['scene_id']]['object_id']
        if row['success']:
            assert row['executed'] and row['physics_evidence_verified']
        if not row['executed']:
            assert row['success'] is False

    recomputed = []
    for group in summary['groups']:
        subset = [r for r in rows if all(r[k] == group[k] for k in ['dataset', 'method', 'platform'])]
        by_object = collections.defaultdict(list)
        for row in subset:
            by_object[row['object_id']].append(row)
        assert all(len(rs) == 12 for rs in by_object.values())
        successes = sum(r['success'] for r in subset)
        covered = sum(any(r['success'] for r in rs) for rs in by_object.values())
        executed = sum(r['executed'] for r in subset)
        assert len(subset) == group['expected_opportunities'] == group['completed_opportunities']
        assert successes == group['successes'] and len(subset)-successes == group['failures']
        assert covered == group['covered'] and len(by_object) == group['objects']
        assert executed == group['execution_count']
        assert len(subset)-executed == group['no_path_count']
        close(successes / len(subset), group['attempt_success_rate_final'])
        close(covered / len(by_object), group['coverage_final'])
        recomputed.append({k: group[k] for k in ['dataset', 'method', 'platform', 'objects', 'covered',
                                                     'expected_opportunities', 'successes', 'execution_count']})

    index = {(r['scene_id'], r['method'], r['platform']): r for r in rows}
    pvalues = []
    for pair in summary['paired_differences']:
        dataset, platform, method, baseline = (pair[k] for k in ['dataset', 'platform', 'method', 'baseline'])
        objects = {v['object_id'] for v in scenes.values() if v['dataset'] == dataset}
        different = collections.Counter()
        for obj in objects:
            ss = [k for k,v in scenes.items() if v['object_id'] == obj]
            ma = any(index[(s,method,platform)]['success'] for s in ss)
            ba = any(index[(s,baseline,platform)]['success'] for s in ss)
            different[(ma,ba)] += 1
        b,c = different[(True,False)],different[(False,True)]
        assert b == pair['object_cad_only_success'] and c == pair['object_baseline_only_success']
        p = min(1.0, 2*sum(math.comb(b+c, k) for k in range(min(b,c)+1))/(2**(b+c))) if b+c else 1.0
        close(p, pair['object_mcnemar_exact_p'])
        pvalues.append(p)
    ordered = sorted(range(len(pvalues)), key=lambda i:pvalues[i])
    current = 1.0
    for rank in range(len(ordered),0,-1):
        i = ordered[rank-1]
        current = min(current, pvalues[i]*len(ordered)/rank)
        close(current, summary['paired_differences'][i]['object_mcnemar_bh_q'])

    geometry = [json.loads(line) for line in (folder/'geometry.jsonl').read_text().splitlines()]
    gkeys = [(r['scene_id'],r['method']) for r in geometry]
    assert len(gkeys) == len(set(gkeys)), 'Duplicate platform-independent geometry'
    assert len(geometry) == (5568 if name.startswith('comparison') else 1392)
    for row in geometry:
        assert row['scene_id'] in scenes

    return rows, {'status':'passed','outcomes':len(rows),'objects':len({s['object_id'] for s in scenes.values()}),
                  'scenes':len(scenes),'methods':len(summary['methods']),'platforms':len(summary['platforms']),
                  'successes':sum(r['success'] for r in rows),'failures':sum(not r['success'] for r in rows),
                  'executed':sum(r['executed'] for r in rows),'geometry_rows':len(geometry),
                  'paired_tests_verified':len(pvalues),'groups':recomputed}


def main():
    scenes = {r['scene_id']:r for r in load(ROOT/'data/scene_index.json')}
    assert len(scenes) == 696
    counts = collections.Counter(r['object_id'] for r in scenes.values())
    assert len(counts) == 58 and set(counts.values()) == {12}
    for sid in scenes:
        for name in ['scene_spec', 'task', 'cameras']:
            assert (ROOT/'data/scenes'/sid/(name+'.json')).is_file()
    primary, a = verify_study('comparison_seed0_v1', scenes)
    ablation, b = verify_study('rq3_source_ablation_seed0_v1', scenes)
    index = {r['task_id']:r for r in primary}
    reused = [r for r in ablation if r['method'] in ['no_cad','cad_fused']]
    assert len(reused) == 2784
    for row in reused:
        original = index[row['task_id']]
        for key,value in original.items():
            if key != 'reused':
                assert row[key] == value, ('Reuse mismatch',row['task_id'],key)
    for entry in load(ROOT/'data/release_provenance.json')['files']:
        p = ROOT/entry['release_path']
        assert p.stat().st_size == entry['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256'], entry['release_path']
    result = {'status':'passed','scope':'Released-row completeness, metric recalculation, exact paired tests, BH adjustment, reuse identity and published file hashes. No new physical execution or raw trajectory replay.',
              'comparison_seed0_v1':a,'rq3_source_ablation_seed0_v1':b,'shared_outcomes_identical':len(reused),
              'unique_outcomes_across_studies':len(primary)+len(ablation)-len(reused)}
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
