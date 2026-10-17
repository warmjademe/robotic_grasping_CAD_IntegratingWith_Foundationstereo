"""Any-success object coverage, keeping environments and incomplete cells separate."""
from itertools import product


def summarize(manifest, outcomes, methods, platforms, seeds):
    for values in [methods, platforms, seeds]:
        if not values or len(set(values)) != len(values):
            raise ValueError("Methods, environments and seeds must be nonempty and unique")
    scenes = {s["scene_id"]: s for s in manifest}
    if not scenes or len(scenes) != len(manifest):
        raise ValueError("Scene manifest must be nonempty and unique")
    expected = set(product(scenes, methods, platforms, seeds))
    observed = {}
    for row in outcomes:
        key = (row["scene_id"], row["method"], row["platform"], row["seed"])
        if key not in expected or key in observed:
            raise ValueError("Unexpected or duplicate result")
        if row["success"] is not None and type(row["success"]) is not bool:
            raise ValueError("Success must be a bool or pending None")
        if row["success"] is True and (row.get("physics_evidence_verified") is not True or not row.get("evidence")):
            raise ValueError("Physical success requires verified evidence")
        observed[key] = row
    objects = sorted({(s["dataset"], s["object_id"]) for s in manifest})
    rows = []
    for (dataset, identity), method, platform in product(objects, methods, platforms):
        ids = [sid for sid, scene in scenes.items() if scene["object_id"] == identity and scene["dataset"] == dataset]
        cells = [observed.get((sid, method, platform, seed)) for sid, seed in product(ids, seeds)]
        successful = [r for r in cells if r and r["success"] is True]
        pending = sum(r is None or r["success"] is None for r in cells)
        covered = True if successful else (None if pending else False)
        rows.append(dict(dataset=dataset, object_id=identity, method=method, platform=platform,
                         covered=covered, successes=len(successful), pending=pending, budget=len(cells),
                         success_witnesses=[dict(scene_id=r["scene_id"], seed=r["seed"], evidence=r.get("evidence")) for r in successful]))
    groups = []
    for dataset, method, platform in product(sorted({x[0] for x in objects}), methods, platforms):
        selected = [r for r in rows if (r["dataset"], r["method"], r["platform"]) == (dataset, method, platform)]
        covered = sum(r["covered"] is True for r in selected)
        unresolved = sum(r["covered"] is None for r in selected)
        pending = sum(r["pending"] for r in selected)
        groups.append(dict(dataset=dataset, method=method, platform=platform, objects=len(selected),
                           covered=covered, uncovered=sum(r["covered"] is False for r in selected),
                           unresolved=unresolved, coverage_lower=covered/len(selected),
                           coverage_upper=(covered+unresolved)/len(selected),
                           coverage_final=None if pending else covered/len(selected),
                           pending_outcomes=pending, experiment_complete=pending == 0))
    return dict(expected_outcomes=len(expected), observed_rows=len(observed),
                completed_outcomes=sum(r["success"] is not None for r in observed.values()),
                per_object=rows, groups=groups)
