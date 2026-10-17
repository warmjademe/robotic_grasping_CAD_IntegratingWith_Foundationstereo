"""Plan frozen predicted grasps with the same Panda geometry for both engines."""
import argparse
import json
import os
import socket
import time
import numpy as np
from common import RUN, STUDY, PROTOCOL, sha, write
from franka_kinematics import Panda, Collision, HOME, DATASET_TO_ROBOT


def plan(task):
    started = time.perf_counter()
    cfg = PROTOCOL['planning']
    budget = cfg['budget_seconds']
    cloud, grasps = STUDY/task['cloud'], STUDY/task['grasps']
    assert sha(cloud) == task['cloud_sha256']
    assert sha(grasps) == task['grasps_sha256']
    with np.load(cloud, allow_pickle=False) as d:
        target, scene = d['target_points_m'].astype(float), d['scene_points_m'].astype(float)
    with np.load(grasps, allow_pickle=False) as d:
        poses, scores, widths, valid = [d[k] for k in ['T_world_grasp','score','jaw_width_m','gripper_width_valid']]
    r = dict(task_id=task['task_id'], status='no_plan', selected=None, rejected=[],
        host=socket.gethostname(), pid=os.getpid(), candidate_count=len(scores),
        ground_truth_used=False, protocol_sha256=sha(RUN/'protocols/protocol.json'),
        cloud_sha256=task['cloud_sha256'], grasps_sha256=task['grasps_sha256'])
    if not len(scores):
        r.update(status='no_candidates', seconds=time.perf_counter()-started)
        return r
    robot = Panda()
    collision = Collision(robot, task['condition'], scene)
    r['home_check'] = collision.check(HOME, .08)
    if r['home_check']['collision']:
        r.update(status='home_collision', seconds=time.perf_counter()-started)
        return r
    from candidate_order import ordered_candidates
    eligible,rejected,total=ordered_candidates(poses,scores,widths,valid,cfg)
    r['rejected'].extend(rejected);r['direction_and_width_eligible']=total
    r['expensive_candidate_budget']=cfg['candidate_limit']
    for rank in eligible:
        rank = int(rank)
        if time.perf_counter()-started >= budget:
            r['status'] = 'planning_timeout'; break
        H, opening = poses[rank].astype(float), float(widths[rank])
        reason = None
        if not valid[rank] or not np.isfinite(opening) or not cfg['opening_range_m'][0] <= opening <= cfg['opening_range_m'][1]:
            reason = 'width_invalid'
        elif H[2,2] > -np.cos(np.deg2rad(cfg['approach_cone_degrees'])):
            reason = 'approach_direction'
        if not reason:
            p = (target-H[:3,3]) @ H[:3,:3]
            band = (np.abs(p[:,0]) < .010) & (p[:,2] > .090) & (p[:,2] < .113)
            if band.sum() < 10: reason = 'no_observed_pad_cross_section'
            else:
                low, high = np.quantile(p[band,1], [.01,.99])
                if not low < 0 < high or max(abs(low),abs(high)) > opening/2:
                    reason = 'observed_cross_section_outside_opening'
        if reason:
            r['rejected'].append(dict(rank=rank, reason=reason)); continue
        hand = DATASET_TO_ROBOT @ H
        pre = hand.copy(); pre[:3,3] -= cfg['approach_distance_m'] * hand[:3,2]
        lift = hand.copy(); lift[2,3] += cfg['lift_distance_m']
        q, error = robot.solve(pre, HOME)
        if error[0] > .0002 or error[1] > np.deg2rad(.1):
            r['rejected'].append(dict(rank=rank,reason='pregrasp_ik')); continue
        paths = dict(transit=[], approach=[], lift=[])
        count = max(2, int(np.ceil(np.rad2deg(np.max(abs(q-HOME)))/cfg['joint_sample_degrees']))+1)
        check = None
        for waypoint in np.linspace(HOME, q, count):
            if time.perf_counter()-started >= budget: reason='planning_timeout'; break
            check = collision.check(waypoint, opening)
            if check['collision']: reason='transit_collision'; break
            paths['transit'].append(waypoint.tolist())
        if not reason:
            for phase, a, b in [('approach',pre,hand),('lift',hand,lift)]:
                count = int(np.ceil(np.linalg.norm(b[:3,3]-a[:3,3])/cfg['cartesian_sample_m']))+1
                for f in np.linspace(0,1,count):
                    if time.perf_counter()-started >= budget: reason='planning_timeout'; break
                    T=a.copy(); T[:3,3]=(1-f)*a[:3,3]+f*b[:3,3]
                    q,error=robot.solve(T,q)
                    if error[0] > .0002 or error[1] > np.deg2rad(.1): reason=phase+'_ik'; break
                    check=collision.check(q,opening)
                    if check['collision']: reason=phase+'_collision'; break
                    paths[phase].append(q.tolist())
                if reason: break
        if reason:
            r['rejected'].append(dict(rank=rank,reason=reason,collision=check))
            if reason == 'planning_timeout': r['status']=reason; break
            continue
        r.update(status='sampled_path_found',selected=dict(rank=rank,score=float(scores[rank]),
            hand_pose=hand.tolist(),hand_pose_dataset_m=H.tolist(),opening=opening,
            closing=cfg['closing_command_m'],estimated_object_width_m=float(high-low),
            closing_policy='Zero-gap position target with 20 N actuator effort limit; object contacts stop the fingers',
            observed_pad_limits_m=[float(low),float(high)],paths=paths))
        break
    r['seconds']=time.perf_counter()-started
    if r['seconds'] > budget:
        r.update(status='planning_timeout',selected=None)
    return r


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='*')
    a=parser.parse_args()
    tasks=json.loads((RUN/'inputs/manifest.json').read_text())['tasks']
    for task in tasks:
        if a.tasks and task['task_id'] not in a.tasks: continue
        dest=RUN/'results/plans'/f"{task['task_id']}.json"
        if dest.exists(): continue
        result=plan(task)
        write(dest,result)
        print(json.dumps({k:result[k] for k in ['task_id','status','seconds']}),flush=True)
