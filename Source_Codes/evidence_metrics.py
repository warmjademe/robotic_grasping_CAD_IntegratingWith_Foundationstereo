"""Evidence audit and fixed-denominator metrics for selected whole-round reproductions.

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
    protocol = read(run / "protocols/protocol.json")
    require(type(task["seed"]) is int and task["seed"] in protocol["comparison"]["seeds"], "Unregistered seed")
    require(task["arm"] in protocol["comparison"]["methods"], "Unexpected comparison arm: " + task["arm"])
    require(bool(task["platforms"]) and set(task["platforms"]) <= set(protocol["platforms"]), "Unregistered platform")
    require((task["perception"], task["grasp"]) == method_parts(task["arm"]), "Arm/backend mismatch")
    base = run / "results/grasps" / tid
    generation_path = base / "task_result.json"
    generation = read(generation_path)
    require(generation["task_id"] == tid, "Candidate task identity mismatch")
    count = generation["returned"]
    require(type(count) is int and 0 <= count <= 400, "Invalid candidate count")
    require(generation.get("seed") == task["seed"], "Candidate record has wrong/missing seed")
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
        require(info.get("seed", info.get("effective_seed")) == task["seed"], "Candidate generation has wrong/missing effective seed")
        if "requested_seed" in info:
            require(info["requested_seed"] == task["seed"], "Candidate generation has wrong requested seed")
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
                perception=task["perception"], grasp=task["grasp"], seed=task["seed"],
                platforms=task["platforms"], release_reproduction=True,
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
    frozen_path = run / "protocols/task_identity.json"
    frozen = {t["task_id"]: t for t in read(frozen_path)}
    require(sha(frozen_path) == read(run / "ops/prepared.json")["task_identity_sha256"], "Changed task registration")
    require(set(frozen) == {t["task_id"] for t in tasks}, "Changed task universe")
    for task in tasks:
        require(all(task.get(k) == v for k, v in frozen[task["task_id"]].items()), "Changed task seed/platform registration")
    prepared = [audited_task(run, task) for task in tasks]
    # Import lazily: aggregation needs neither simulation nor geometry packages.
    from audit_physical_records import audit
    audit(run)
    audit_path = run / "results/physical_trace_audit.json"
    physical_audit = read(audit_path)
    require(physical_audit.get("passed") is True and not physical_audit.get("calibration_only"), "Physical audit incomplete")
    evidence = {(r["platform"], r["task_id"]): r for r in physical_audit["rows"] if not r["calibration"]}
    expected = {(p, r["candidate_task_id"]) for r in prepared if r["planning_status"] == "sampled_path_found" for p in r["platforms"]}
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
        for platform in common["platforms"]:
            row = dict(common, platform=platform,
                       task_id=f"{common['scene_id']}__{common['method']}__{platform}__seed{common['seed']}",
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
    result = dict(status="audited", rows=rows, expected_rows=sum(len(t["platforms"]) for t in tasks),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('geometry', 'audit'))
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = geometry_batch(args.run) if args.command == 'geometry' else audit_batch(args.run)
    print(args.command, len(result['rows']), 'records')
