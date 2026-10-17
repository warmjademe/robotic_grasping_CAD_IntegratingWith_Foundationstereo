"""Aggregate geometry once per stereo scene, independent of grasp seed/engine."""
from collections import defaultdict


def groups_for(rows):
    grouped=defaultdict(list);seen=set()
    for row in rows:
        key=(row['scene_id'],row['method'])
        if key in seen:raise ValueError('Repeated geometry scene/method')
        seen.add(key)
        grouped[(row['dataset'],row['method'])].append(row)
        grouped[('all',row['method'])].append(row)
    groups=[]
    for (dataset,method),selected in sorted(grouped.items()):
        errors=[r['surface_mae_mm'] for r in selected if r['surface_mae_mm'] is not None]
        groups.append(dict(dataset=dataset,method=method,scenes=len(selected),objects=len({r['object_id'] for r in selected}),
            empty_scenes=sum(r['points']==0 for r in selected),
            surface_mae_mm_nonempty=sum(errors)/len(errors) if errors else None,
            precision_2mm=sum(r['precision_2mm'] for r in selected)/len(selected),
            complete_coverage_2mm=sum(r['complete_coverage_2mm'] for r in selected)/len(selected)))
    return groups


# CLI aggregation is provided by single_run_metrics.py with an explicit --study.
