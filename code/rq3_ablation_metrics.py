"""Apply unchanged physical checks to separate RQ3 arms; preserve full denominators."""
import argparse
from pathlib import Path
from collections import Counter

from project_common import ROOT, read, write, sha
import single_run_metrics as base

NEW_METHODS = ('registered_cad__graspgen', 'direct_concat__graspgen')
ALL_METHODS = ('no_cad', 'cad_fused') + NEW_METHODS
base.METHODS = ALL_METHODS


def aggregate(study):
    scenes = read(study / 'datasets/manifest.json')['scenes']
    reused_path = study / 'results/reused_outcomes.json'
    reused = read(reused_path)
    rows, sources, gates = list(reused['rows']), {str(reused_path.relative_to(ROOT)): sha(reused_path)}, {}
    geometry = []
    for item in read(study / 'queue.json')['runs']:
        run = ROOT / item['path']
        path = run / 'results/outcomes.json'
        if path.exists():
            data = read(path)
            assert data['status'] == 'audited'
            rows += data['rows']
            sources[str(path.relative_to(ROOT))] = sha(path)
        geom = run / 'results/geometry.json'
        if geom.exists():
            geometry += read(geom)['rows']
        for p in (run / 'results/perception/registered_cad').glob('*/perception.json'):
            d = read(p)
            gates[d['scene_id']] = dict(accepted=d['accepted'], reasons=d['gate_failure_reasons'])
    summary, outcomes = base.summary_metrics(scenes, rows, methods=ALL_METHODS)
    summary.update(new_expected_outcomes=2784, reused_outcomes=2784,
        new_completed_outcomes=sum(r['method'] in NEW_METHODS and r['success'] is not None for r in outcomes),
        source_hashes=sources, source_sha256=sha(__file__), original_main_comparison_unchanged=True,
        gate_counts=dict(Counter('accepted' if d['accepted'] else 'rejected' for d in gates.values())),
        gate_reasons=dict(Counter(reason for d in gates.values() for reason in d['reasons'])),
        gate_scenes=gates, geometry=geometry)
    eligible = [s for s in scenes if gates.get(s['scene_id'], {}).get('accepted')]
    if eligible:
        ids = {s['scene_id'] for s in eligible}
        subset = [r for r in rows if r['scene_id'] in ids]
        # Avoid presenting a reduced-scene object coverage as full 12-scene coverage.
        summary['accepted_scene_subset'] = dict(condition='Frozen v6 global gate passed; M actually fused',
            scene_count=len(eligible), object_count=len({s['object_id'] for s in eligible}),
            groups=[dict(dataset=d, platform=p, method=m, scenes=sum(s['dataset']==d for s in eligible),
                completed=sum(r['success'] is not None for r in subset if (r['dataset'],r['platform'],r['method'])==(d,p,m)),
                successes=sum(r['success'] is True for r in subset if (r['dataset'],r['platform'],r['method'])==(d,p,m)))
                for d in ('tless','itodd') for p in base.PLATFORMS for m in ALL_METHODS])
    write(study / 'results/summary.json', summary)
    write(study / 'results/outcomes.json', dict(status=summary['status'], expected_outcomes=5568,
        rows=outcomes, sources=sources, source_sha256=sha(__file__)))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit','geometry','aggregate'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--study', type=Path)
    args=parser.parse_args()
    if args.command == 'audit':
        result=base.audit_batch(args.run)
        result['rq3_adapter_sha256']=sha(__file__)
        result['formal_main_comparison']=False
        for row in result['rows']:
            row['formal_result']=False
            row['study_scope']='exploratory_rq3_source_ablation'
        write(args.run / 'results/outcomes.json', result)
        print('RQ3_AUDITED',len(result['rows']),flush=True)
    elif args.command == 'geometry':
        result=base.geometry_batch(args.run)
        result['formal_test']=False
        write(args.run / 'results/geometry.json',result)
        print('RQ3_GEOMETRY',len(result['rows']),flush=True)
    else:
        result=aggregate(args.study)
        print('RQ3_NEW_PROGRESS',result['new_completed_outcomes'],'/ 2784',flush=True)
