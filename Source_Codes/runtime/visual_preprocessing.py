"""Shared RGB-only target selection, including low objects near the table."""
import numpy as np
from scipy.spatial import cKDTree
from geometry_v6 import foreground_confidence


def select_target(points, foreground, roi, table_height, cfg):
    points = np.asarray(points, dtype=float)
    foreground = np.asarray(foreground, dtype=float)
    if len(points) != len(foreground):
        raise ValueError("Point and foreground arrays differ")
    lo = np.asarray(roi["min"], dtype=float).copy()
    hi = np.asarray(roi["max"], dtype=float)
    lo[2] = table_height - cfg["observation_table_tolerance_m"]
    selected = ((points >= lo) & (points <= hi)).all(axis=1)
    selected &= foreground >= cfg["minimum_foreground_confidence"]
    target = points[selected]
    raw = len(target)
    k = cfg["knn_neighbors"]
    if len(target) >= k:
        distance = cKDTree(target).query(target, k=k)[0][:, 1:].mean(axis=1)
        target = target[distance <= distance.mean() + cfg["knn_standard_deviations"] * distance.std()]
    return target, raw


def export_cloud(out, depth, valid, images, calibration, task, source_hashes, route="stereo"):
    from contract import save
    import json
    from pathlib import Path
    cfg = json.loads((Path(__file__).resolve().parent / "preprocessing_v1.json").read_text())
    camera = calibration["cameras"]["left"]
    K, T = np.asarray(camera["K"]), np.asarray(camera["world_from_camera"])
    y, x = np.indices(depth.shape)
    z = depth[valid]
    points = np.column_stack(((x[valid]-K[0,2])*z/K[0,0], (y[valid]-K[1,2])*z/K[1,1], z))
    points = points @ T[:3,:3].T + T[:3,3]
    colors = images[0][valid]
    foreground, record = foreground_confidence(images[0], camera, task["target_roi_world_m"],
        task["known_table_height_m"], cfg["foreground"])
    confidence = foreground[valid]
    workspace = ((points >= cfg["workspace_min_m"]) & (points <= cfg["workspace_max_m"])).all(axis=1)
    points, colors, confidence = points[workspace], colors[workspace], confidence[workspace]
    target, raw = select_target(points, confidence, task["target_roi_world_m"], task["known_table_height_m"], cfg)
    np.savez_compressed(out / "cloud.npz", scene_points_m=points.astype(np.float32),
        target_points_m=target.astype(np.float32), scene_colors=colors)
    np.savez_compressed(out / "target_selection.npz", foreground_confidence=foreground)
    metadata = dict(schema_version=1, units="m", frame_id="dataset_world", input_hashes=source_hashes,
        purpose="grasp_inference", geometry_source="perception_prediction", renderer_depth_used=False,
        object_pose_used=False, cad_used=False, reference_convention="opencv_optical_x_right_y_down_z_forward",
        scene_scope="workspace_roi", gripper_id="franka_panda", perception_route=route,
        gripper_geometry=dict(max_width_m=.08, closing_axis="+y", approach_axis="+z", tcp_frame="panda_hand"),
        has_normals=False, has_colors=True, target_selection=cfg["version"], foreground_estimate=record)
    ready = min(len(points), len(target)) > 128
    if ready:
        save(out / "grasp_input.npz", dict(scene_points_m=points.astype(np.float32),
            target_points_m=target.astype(np.float32), scene_colors=colors, camera_origins_m=T[None,:3,3],
            T_world_reference=T, point_camera_ids=np.zeros(len(points),dtype=np.int32)), metadata)
    (out / "selection.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return dict(scene_points=len(points), raw_target_points=raw, target_points=len(target), grasp_input_ready=ready)
