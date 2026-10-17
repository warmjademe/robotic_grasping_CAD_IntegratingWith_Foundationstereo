"""Evidence audit and fixed-denominator metrics for the seed-zero comparison.

This module never generates candidates or runs physics. Missing results remain
pending; one successful scene establishes object coverage without ending its
remaining scene budget. Geometry is evaluated once per perception and scene.
"""
import argparse
import csv
import json
from collections import Counter
from itertools import product
from math import comb
from pathlib import Path

from project_common import ROOT, read, sha, write
from coverage_metrics import summarize

PERCEPTIONS = ("foundationstereo", "stereobm", "sgm", "psmnet", "gwcnet", "raft_stereo", "igev_stereo")
OTHER_GRASPS = ("6dof_graspnet", "contact_graspnet", "gpd", "graspnet_baseline")
METHODS = ("no_cad", "cad_fused") + tuple(p + "__graspgen" for p in PERCEPTIONS[1:]) + tuple(
    "foundationstereo__" + g for g in OTHER_GRASPS)
PLATFORMS = ("franka_isaac", "franka_mujoco")
TERMINAL_NO_PATH = {"no_candidates", "insufficient_points", "no_plan", "home_collision",
                    "planning_timeout", "algorithm_timeout", "invalid_prediction", "upstream_algorithm_failure"}
PHYSICAL_FIELDS = ("physics_lift_success", "approach_endpoint_reached", "collision_free",
                   "execution_and_return", "returned_to_origin_success", "prohibited_contact_steps",
                   "max_arm_contact_force_n", "max_finger_non_target_force_n", "approach_position_error_mm",
                   "approach_rotation_error_deg", "min_hold_bottom_clearance_m", "return_xy_error_m",
                   "final_bottom_clearance_m", "hold_sim_s")


def pointer(path, root=ROOT):
    """Use project-relative pointers where possible, preserving external roots."""
    path, root = Path(path).absolute(), Path(root).absolute()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def method_parts(method):
    if method == "no_cad":
        return "foundationstereo", "graspgen"
    if method == "cad_fused":
        return "cad_fused", "graspgen"
    return tuple(method.split("__", 1))


def verify_hash(path, expected, label):
    require(bool(expected), label + " lacks a recorded hash")
    actual = sha(path)
    require(actual == expected, label + " hash mismatch: " + str(path))
    return actual


def audited_task(run, task):
    """Validate each prediction/plan chain, including zero-candidate outcomes."""
    tid = task["task_id"]
    require(type(task["seed"]) is int and task["seed"] == 0, "Only seed zero is allowed")
    require(task["arm"] in METHODS, "Unexpected comparison arm: " + task["arm"])
    require((task["perception"], task["grasp"]) == method_parts(task["arm"]), "Arm/backend mismatch")
    base = run / "results/grasps" / tid
    generation_path = base / "task_result.json"
    generation = read(generation_path)
    require(generation["task_id"] == tid, "Candidate task identity mismatch")
    count = generation["returned"]
    require(type(count) is int and 0 <= count <= 400, "Invalid candidate count")
    require(generation.get("seed", 0) == 0, "Candidate record has wrong seed")
    cloud_path = run / task.get("cloud", f"results/perception/{task['perception']}/{task['episode_id']}/cloud.npz")
    cloud_hash = verify_hash(cloud_path, generation.get("input_sha256"), "Candidate input")
    if task.get("cloud_sha256"):
        require(task["cloud_sha256"] == cloud_hash, "Manifest cloud hash mismatch")
    perception_path = cloud_path.with_name("perception.json")
    perception = read(perception_path)
    require(perception.get("cloud_sha256") == cloud_hash, "Perception cloud hash mismatch")
    require(not perception.get("geometry_truth_used") and not perception.get("object_pose_used"), "Evaluation truth used as input")
    plan_path = run / "results/plans" / (tid + ".json")
    plan = read(plan_path)
    require(plan["task_id"] == tid, "Plan identity mismatch")
    require(plan["status"] == "sampled_path_found" or plan["status"] in TERMINAL_NO_PATH,
            "Unfinished/infrastructure plan status: " + plan["status"])
    info = {}
    candidate_hash = None
    if count or (base / "grasp_generation.json").exists():
        info = read(base / "grasp_generation.json")
        require(info.get("seed", info.get("effective_seed")) == 0, "Candidate generation has wrong/missing effective seed")
        if "requested_seed" in info:
            require(info["requested_seed"] == 0, "Candidate generation has wrong requested seed")
        require(info["returned"] == count, "Candidate count mismatch")
        candidate_path = run / task.get("grasps", f"results/grasps/{tid}/grasps.npz")
        candidate_hash = verify_hash(candidate_path, info.get("output_sha256"), "Candidates")
        require(generation.get("grasps_sha256") == candidate_hash, "Candidate receipt hash mismatch")
        if task.get("grasps_sha256"):
            require(task["grasps_sha256"] == candidate_hash, "Manifest candidate hash mismatch")
        verify_hash(cloud_path.with_name("grasp_input.npz"), info.get("input_npz_sha256"), "Grasp adapter input")
    if count:
        require(generation["status"] == "completed", "Returned candidates lack completed receipt")
        require(plan.get("ground_truth_used") is False, "Plan truth-use declaration missing/invalid")
        require(plan.get("cloud_sha256") == cloud_hash and plan.get("grasps_sha256") == candidate_hash, "Plan input hash mismatch")
        verify_hash(run / "protocols/protocol.json", plan.get("protocol_sha256"), "Planning protocol")
    else:
        require(generation["status"] in TERMINAL_NO_PATH, "Zero candidates from infrastructure error/unresolved task")
        require(plan["status"] != "sampled_path_found", "Zero-candidate task cannot have a path")
    return dict(candidate_task_id=tid, scene_id=task["episode_id"], object_id=task["object_id"],
                dataset=task["dataset"], condition=task["condition"], method=task["arm"],
                perception=task["perception"], grasp=task["grasp"], seed=0, formal_result=True,
                planning_status=plan["status"], generation_status=generation["status"], candidates=count,
                eligible_candidates=plan.get("direction_and_width_eligible"),
                selected_rank=(plan.get("selected") or {}).get("rank"),
                generation_seconds=generation.get("seconds", info.get("seconds")),
                planning_seconds=plan.get("seconds"), perception_seconds=perception.get("seconds"),
                perception_inference_seconds=perception.get("inference_seconds"),
                cloud_sha256=cloud_hash, grasps_sha256=candidate_hash,
                candidate_receipt=pointer(generation_path), candidate_receipt_sha256=sha(generation_path),
                plan=pointer(plan_path), plan_sha256=sha(plan_path),
                perception_record=pointer(perception_path), perception_record_sha256=sha(perception_path))


def audit_batch(run):
    run = Path(run).resolve()
    tasks = read(run / "inputs/manifest.json")["tasks"]
    require(bool(tasks) and len({t["task_id"] for t in tasks}) == len(tasks), "Empty/duplicate manifest tasks")
    prepared = [audited_task(run, task) for task in tasks]
    # Import lazily: aggregation needs neither simulation nor geometry packages.
    from audit_physical_records import audit
    audit(run)
    audit_path = run / "results/physical_trace_audit.json"
    physical_audit = read(audit_path)
    require(physical_audit.get("passed") is True and not physical_audit.get("calibration_only"), "Physical audit incomplete")
    evidence = {(r["platform"], r["task_id"]): r for r in physical_audit["rows"] if not r["calibration"]}
    expected = {(p, r["candidate_task_id"]) for r in prepared if r["planning_status"] == "sampled_path_found" for p in PLATFORMS}
    require(set(evidence) == expected, "Missing or unexpected executed physical records")
    receipts = {}
    for file in sorted((run / "ops").glob("offline_*.json")):
        receipt = read(file)
        require(receipt.get("status") == "completed" and receipt.get("network_isolated") is True,
                "Unfinished/non-isolated offline receipt: " + str(file))
        require(receipt.get("access", {}).get("blocked_reads") == 0, "Blocked evaluation input read in offline receipt")
        receipts[pointer(file)] = sha(file)
    rows = []
    reports = {}
    for common in prepared:
        for platform in PLATFORMS:
            row = dict(common, platform=platform,
                       task_id=f"{common['scene_id']}__{common['method']}__{platform}__seed0",
                       outcome_status="completed", reused=False)
            if common["planning_status"] == "sampled_path_found":
                checked = evidence[(platform, common["candidate_task_id"])]
                file = run / checked["result"]
                verify_hash(file, checked["result_sha256"], "Physical result")
                verify_hash(file.with_name("trace.npz"), checked["trace_sha256"], "Physical trace")
                if file not in reports:
                    reports[file] = read(file)
                report = reports[file]
                raw = [r for r in report["rows"] if r["task_id"] == common["candidate_task_id"]]
                require(len(raw) == 1, "Ambiguous raw physical row")
                raw = raw[0]
                require(raw["execution_success"] == checked["success"], "Physical audit/result disagreement")
                row.update(success=checked["success"], executed=True, physics_evidence_verified=True,
                           evidence=pointer(file), result_sha256=checked["result_sha256"],
                           trace=pointer(file.with_name("trace.npz")), trace_sha256=checked["trace_sha256"],
                           physical_metrics={k: raw[k] for k in PHYSICAL_FIELDS if k in raw},
                           simulation_wall_seconds=report.get("simulation_wall_seconds"),
                           physics_report_wall_seconds=report.get("seconds"),
                           physics_report_task_count=len(report["rows"]),
                           physics_timing_scope="whole raw report; shared batch values must not be summed per task")
            else:
                row.update(success=False, executed=False, physics_evidence_verified=False, evidence=None,
                           failure_stage="planning" if common["candidates"] else "candidate_or_perception")
            rows.append(row)
    result = dict(status="audited", seed=0, rows=rows, expected_rows=2 * len(tasks),
                  manifest_sha256=sha(run / "inputs/manifest.json"),
                  protocol_sha256=sha(run / "protocols/protocol.json"),
                  physical_trace_audit_sha256=sha(audit_path), offline_receipts=receipts,
                  code_sha256=sha(__file__))
    write(run / "results/outcomes.json", result)
    return result


def geometry_batch(run):
    import numpy as np
    from scipy.spatial import cKDTree
    run = Path(run).resolve()
    tasks = read(run / "inputs/manifest.json")["tasks"]
    requested = {(t["episode_id"], t["perception"]) for t in tasks}
    imports_path = run / "ops/imports.json"
    shared_primary = read(imports_path).get("shared_from") if imports_path.exists() else None
    rows, unavailable = [], []
    scenes = read(run / "datasets/manifest.json")["scenes"]
    for scene in scenes:
        sid, oid = scene["scene_id"], scene["object_id"]
        selected = {p for s, p in requested if s == sid}
        # Reused FS/CAD clouds remain useful even when their tasks were excluded.
        selected.update(p for p in ("foundationstereo", "cad_fused")
                        if (run / "results/perception" / p / sid / "cloud.npz").exists())
        if shared_primary:
            # Its prerequisite primary run already evaluates these exact clouds.
            selected.difference_update(("foundationstereo", "cad_fused"))
        truth_path = run / "datasets/evaluation" / sid / "scene.json"
        target = next(x for x in read(truth_path)["instances"] if x["role"] == "target")
        transform = np.asarray(target["world_from_object"])
        surface_path = run / "datasets/assets" / oid / "cad_samples.npz"
        with np.load(surface_path) as data:
            surface = data["points_m"] @ transform[:3, :3].T + transform[:3, 3]
        for perception in sorted(selected):
            file = run / "results/perception" / perception / sid / "cloud.npz"
            if not file.exists():
                unavailable.append(dict(scene_id=sid, perception=perception, status="pending"))
                continue
            with np.load(file) as data:
                points = data["target_points_m"]
            require(np.isfinite(points).all(), "Nonfinite cloud: " + str(file))
            distance = cKDTree(surface).query(points)[0] if len(points) else np.array([])
            reverse = cKDTree(points).query(surface)[0] if len(points) else np.full(len(surface), np.inf)
            method = {"foundationstereo": "no_cad", "cad_fused": "cad_fused"}.get(perception, perception + "__graspgen")
            rows.append(dict(scene_id=sid, object_id=oid, dataset=scene["dataset"], condition=scene["condition"],
                             method=method, perception=perception, points=len(points),
                             surface_mae_mm=float(distance.mean() * 1000) if len(points) else None,
                             surface_p95_mm=float(np.percentile(distance, 95) * 1000) if len(points) else None,
                             precision_2mm=float((distance < .002).mean()) if len(points) else 0.,
                             complete_coverage_2mm=float((reverse < .002).mean()),
                             cloud=pointer(file), cloud_sha256=sha(file),
                             reference_surface_sha256=sha(surface_path),
                             evaluation_scene=pointer(truth_path), evaluation_scene_sha256=sha(truth_path)))
    result = dict(formal_test=True, evaluation_only=True, rows=rows, pending=unavailable,
                  code_sha256=sha(__file__),
                  reference="Fixed public CAD surface samples transformed by simulation truth for evaluation only; once per scene/perception")
    write(run / "results/geometry.json", result)
    return result


def complete_rows(scenes, observed, methods=METHODS, platforms=PLATFORMS):
    """Expand only the registered universe; reject duplicates and bad evidence."""
    scene_lookup = {s["scene_id"]: s for s in scenes}
    # Existing coverage helper checks duplicate keys and verified successes.
    coverage = summarize(scenes, observed, methods, platforms, [0])
    lookup = {}
    for raw in observed:
        require(type(raw["seed"]) is int and raw["seed"] == 0, "Only seed zero is allowed")
        scene = scene_lookup[raw["scene_id"]]
        require(raw.get("object_id", scene["object_id"]) == scene["object_id"] and
                raw.get("dataset", scene["dataset"]) == scene["dataset"], "Outcome scene identity mismatch")
        if raw["success"] is not None:
            if raw.get("executed"):
                require(raw.get("physics_evidence_verified") is True and bool(raw.get("evidence")), "Executed result lacks physical audit")
            elif raw.get("planning_status"):
                require(raw["planning_status"] in TERMINAL_NO_PATH and raw["success"] is False,
                        "Unexecuted result lacks terminal no-path status")
        lookup[(raw["scene_id"], raw["method"], raw["platform"])] = raw
    rows = []
    for scene, method, platform in product(scenes, methods, platforms):
        key = (scene["scene_id"], method, platform)
        perception, grasp = method_parts(method) if method in METHODS else (None, None)
        row = dict(scene_id=scene["scene_id"], object_id=scene["object_id"], dataset=scene["dataset"],
                   condition=scene.get("condition"), method=method, platform=platform, seed=0,
                   perception=perception, grasp=grasp,
                   task_id=f"{scene['scene_id']}__{method}__{platform}__seed0",
                   success=None, executed=None, outcome_status="pending")
        row.update(lookup.get(key, {}))
        row["outcome_status"] = "pending" if row["success"] is None else "completed"
        rows.append(row)
    return rows, coverage


def summary_metrics(scenes, observed, methods=METHODS, platforms=PLATFORMS):
    rows, coverage = complete_rows(scenes, observed, methods, platforms)
    groups = []
    numeric_fields = ("generation_seconds", "planning_seconds", "perception_seconds", "candidates", "eligible_candidates")
    for group in coverage["groups"]:
        selected = [r for r in rows if all(r[k] == group[k] for k in ("dataset", "method", "platform"))]
        completed = [r for r in selected if r["success"] is not None]
        successes = sum(r["success"] is True for r in completed)
        pending = len(selected) - len(completed)
        result = dict(group, expected_opportunities=len(selected), completed_opportunities=len(completed),
                      successes=successes, failures=len(completed) - successes,
                      execution_count=sum(r.get("executed") is True for r in completed),
                      no_path_count=sum(r.get("executed") is False for r in completed),
                      attempt_success_lower=successes / len(selected),
                      attempt_success_upper=(successes + pending) / len(selected),
                      attempt_success_rate_final=successes / len(selected) if not pending else None,
                      attempt_success_rate_completed=successes / len(completed) if completed else None,
                      planning_status_counts=dict(Counter(r.get("planning_status", "unknown") for r in completed)))
        result["means_observed"] = {}
        for field in numeric_fields:
            values = [r[field] for r in completed if isinstance(r.get(field), (int, float))]
            result["means_observed"][field] = dict(n=len(values), mean=sum(values) / len(values) if values else None)
        result["physical_means_observed"] = {}
        for field in PHYSICAL_FIELDS:
            values = [r["physical_metrics"][field] for r in completed
                      if isinstance(r.get("physical_metrics", {}).get(field), (int, float))]
            result["physical_means_observed"][field] = dict(n=len(values), mean=sum(values) / len(values) if values else None)
        groups.append(result)
    paired = []
    if "cad_fused" in methods:
        by_key = {(r["scene_id"], r["method"], r["platform"]): r for r in rows}
        for dataset, platform, baseline in product(sorted({s["dataset"] for s in scenes}), platforms,
                                                   [m for m in methods if m != "cad_fused"]):
            relevant = [s for s in scenes if s["dataset"] == dataset]
            pairs = [(by_key[(s["scene_id"], "cad_fused", platform)], by_key[(s["scene_id"], baseline, platform)]) for s in relevant]
            done = [(a, b) for a, b in pairs if a["success"] is not None and b["success"] is not None]
            wins = sum(a["success"] and not b["success"] for a, b in done)
            losses = sum(b["success"] and not a["success"] for a, b in done)
            both = sum(a["success"] and b["success"] for a, b in done)
            objects = [r for r in coverage["per_object"] if r["dataset"] == dataset and r["platform"] == platform]
            ours = {r["object_id"]: r for r in objects if r["method"] == "cad_fused"}
            theirs = {r["object_id"]: r for r in objects if r["method"] == baseline}
            object_done = [(ours[o], theirs[o]) for o in ours if ours[o]["pending"] == theirs[o]["pending"] == 0]
            object_wins = sum(a["covered"] and not b["covered"] for a, b in object_done)
            object_losses = sum(b["covered"] and not a["covered"] for a, b in object_done)
            object_complete = len(object_done) == len(ours)
            paired.append(dict(dataset=dataset, platform=platform, method="cad_fused", baseline=baseline,
                               expected_scene_pairs=len(pairs), completed_scene_pairs=len(done),
                               pending_scene_pairs=len(pairs) - len(done), cad_only_success=wins, baseline_only_success=losses,
                               both_success=both, neither_success=len(done) - wins - losses - both,
                               success_difference_completed_pairs=(wins - losses) / len(done) if done else None,
                               success_difference_final=(wins - losses) / len(pairs) if len(done) == len(pairs) else None,
                               expected_object_pairs=len(ours), completed_object_pairs=len(object_done),
                               pending_object_pairs=len(ours) - len(object_done),
                               object_cad_only_success=object_wins, object_baseline_only_success=object_losses,
                               object_coverage_difference_completed_pairs=sum(int(a["covered"]) - int(b["covered"]) for a, b in object_done) / len(object_done) if object_done else None,
                               object_coverage_difference_final=(object_wins - object_losses) / len(ours) if object_complete else None,
                               object_mcnemar_exact_p=exact_mcnemar(object_wins, object_losses) if object_complete else None,
                               object_mcnemar_bh_q=None,
                               comparison_scope="Object-level exact McNemar only after all paired scene budgets finish; one seed does not estimate repeated-run variance"))
    if paired and all(p["object_mcnemar_exact_p"] is not None for p in paired):
        for pair, q in zip(paired, bh_adjust([p["object_mcnemar_exact_p"] for p in paired])):
            pair["object_mcnemar_bh_q"] = q
    return dict(status="complete" if coverage["completed_outcomes"] == coverage["expected_outcomes"] else "interim",
                seed=0, methods=list(methods), platforms=list(platforms),
                expected_outcomes=coverage["expected_outcomes"], completed_outcomes=coverage["completed_outcomes"],
                pending_outcomes=coverage["expected_outcomes"] - coverage["completed_outcomes"],
                groups=groups, per_object=coverage["per_object"], paired_differences=paired,
                multiple_testing_family="All registered CAD-versus-baseline object-coverage comparisons across both datasets and platforms; BH values pending until the whole family finishes",
                ranking_status="pending" if coverage["completed_outcomes"] != coverage["expected_outcomes"] else "complete_comparison_available",
                primary_metric="At least one successful scene per object, separately for each platform and method",
                completion_rule="All registered scenes run once even after object coverage is established",
                denominator_rule="Missing/incomplete evidence is pending; terminal no-path outcomes are failures",
                uncertainty="Coverage and success bounds include pending outcomes; one seed cannot estimate repeated-run variability"), rows


def exact_mcnemar(first_only, second_only):
    """Two-sided exact conditional binomial test of discordant paired objects."""
    n = first_only + second_only
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(first_only, second_only) + 1)) / 2 ** n)


def bh_adjust(pvalues):
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [None] * len(pvalues)
    last = 1.0
    for rank in range(len(order), 0, -1):
        index = order[rank - 1]
        last = min(last, pvalues[index] * len(order) / rank)
        adjusted[index] = last
    return adjusted


def enrich_reused(row, root, cache):
    """Read optional local metrics without weakening the prior reuse audit.

    Mac checkouts can omit original trajectories. Missing files leave metrics
    absent; present files must match the original audit and are never trusted
    merely because a path exists.
    """
    row = dict(row)
    source = row.get("reused_from")
    if not source:
        return row
    run = (root / source).parent.parent
    def checked(path, digest):
        if not path.exists():
            return None
        key = (str(path), digest)
        if key not in cache:
            verify_hash(path, digest, "Reused metric evidence")
            cache[key] = read(path)
        return cache[key]
    plan_path = run / "results/plans" / (row["candidate_task_id"] + ".json")
    plan = checked(plan_path, row["plan_sha256"])
    if plan:
        row.update(plan=pointer(plan_path, root), planning_seconds=plan.get("seconds"),
                   candidates=plan.get("candidate_count"), eligible_candidates=plan.get("direction_and_width_eligible"),
                   selected_rank=(plan.get("selected") or {}).get("rank"))
    candidate_path = run / "results/grasps" / row["candidate_task_id"] / "task_result.json"
    candidate = checked(candidate_path, row.get("reuse_audit", {}).get("candidate_task_result_sha256"))
    if candidate:
        row.update(generation_seconds=candidate.get("seconds"), candidates=candidate["returned"],
                   generation_status=candidate["status"], candidate_receipt=pointer(candidate_path, root))
    if row.get("evidence"):
        file = root / row["evidence"]
        report = checked(file, row["result_sha256"])
        if report:
            found = [r for r in report["rows"] if r["task_id"] == row["candidate_task_id"]]
            require(len(found) == 1, "Reused physical metric identity mismatch")
            raw = found[0]
            require(raw["execution_success"] == row["success"], "Reused physical metric success mismatch")
            row.update(physical_metrics={k: raw[k] for k in PHYSICAL_FIELDS if k in raw},
                       simulation_wall_seconds=report.get("simulation_wall_seconds"),
                       physics_report_wall_seconds=report.get("seconds"), physics_report_task_count=len(report["rows"]),
                       physics_timing_scope="whole raw report; shared batch values must not be summed per task")
    return row


def load_scenes(study, root):
    candidates = [study / "datasets/manifest.json", study / "scene_manifest.json", root / "datasets/scene_index.json"]
    file = next((p for p in candidates if p.exists()), None)
    require(file is not None, "Missing complete scene manifest")
    manifest = read(file)
    scenes = manifest["scenes"] if isinstance(manifest, dict) else manifest
    require(len(scenes) == 696, "Comparison requires all 696 registered scenes")
    counts = Counter((s["dataset"], s["object_id"]) for s in scenes)
    require(len(counts) == 58 and set(counts.values()) == {12}, "Expected 58 objects with 12 scenes each")
    require(Counter(dataset for dataset, _ in counts) == {"tless": 30, "itodd": 28}, "Dataset object counts changed")
    return scenes, file


def aggregate(study, root=ROOT):
    study, root = Path(study).resolve(), Path(root).resolve()
    scenes, scene_file = load_scenes(study, root)
    observed, sources, metric_cache = [], {}, {}
    reused_file = study / "results/reused_outcomes.json"
    files = ([reused_file] if reused_file.exists() else []) + sorted((study / "batches").glob("*/results/outcomes.json"))
    for file in files:
        data = read(file)
        require(data.get("status") in {"audited", "audited_reuse", "verified", "reuse_verified", "reused"} or
                (file == reused_file and data.get("passed") is True), "Source outcomes not audited: " + str(file))
        digest = sha(file)
        sources[pointer(file, root)] = digest
        for row in data["rows"]:
            require(type(row.get("success")) is bool, "Unfinished row in audited outcome source")
            if file == reused_file:
                row = enrich_reused(row, root, metric_cache)
            observed.append(dict(row, reused=file == reused_file,
                                 outcome_source=pointer(file, root), outcome_source_sha256=digest))
    summary, rows = summary_metrics(scenes, observed)
    summary.update(scene_manifest=pointer(scene_file, root), scene_manifest_sha256=sha(scene_file),
                   sources=sources, code_sha256=sha(__file__), reused_outcomes=sum(r.get("reused") is True for r in rows))
    geometry_rows, geometry_sources, geometry_seen = [], {}, set()
    for file in sorted((study / "batches").glob("*/results/geometry.json")):
        data = read(file)
        require(data.get("evaluation_only") is True, "Geometry must be evaluation-only")
        geometry_sources[pointer(file, root)] = sha(file)
        for row in data["rows"]:
            key = (row["scene_id"], row.get("perception", row["method"]))
            require(key not in geometry_seen, "Duplicate geometry scene/perception")
            geometry_seen.add(key)
            geometry_rows.append(row)
    if geometry_rows:
        from summarize_main_geometry import groups_for
        summary["geometry"] = dict(rows=geometry_rows, groups=groups_for(geometry_rows), sources=geometry_sources,
                                   scope="Once per scene/perception, independent of grasp strategy and platform")
    write(study / "results/summary.json", summary)
    write(study / "results/outcomes.json", dict(status=summary["status"], expected_outcomes=len(rows), rows=rows,
                                               sources=sources, code_sha256=sha(__file__)))
    path = study / "results/outcomes.csv"
    fields = sorted({k for row in rows for k in row})
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    temporary.replace(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "geometry"):
        commands.add_parser(name).add_argument("--run", type=Path, required=True)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--study", type=Path, required=True)
    aggregate_parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_batch(args.run)
        print("SINGLE_RUN_OUTCOMES_AUDITED", len(result["rows"]), flush=True)
    elif args.command == "geometry":
        result = geometry_batch(args.run)
        print("SINGLE_RUN_GEOMETRY", len(result["rows"]), "pending", len(result["pending"]), flush=True)
    else:
        result = aggregate(args.study, args.root)
        print("SINGLE_RUN_PROGRESS", result["completed_outcomes"], "/", result["expected_outcomes"], result["status"], flush=True)


if __name__ == "__main__":
    main()
