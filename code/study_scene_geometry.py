"""Deterministic scene construction from registered seeds and known CAD geometry."""
import copy
import itertools
import numpy as np


def transform(bounds,T):
    corners=np.array(list(itertools.product(*zip(bounds[0],bounds[1]))))
    points=corners@T[:3,:3].T+T[:3,3]
    return np.array([points.min(axis=0),points.max(axis=0)])


def pose(x,y,yaw,clearance):
    c,s=np.cos(yaw),np.sin(yaw);T=np.eye(4)
    T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=[x,y,clearance]
    return T


def make_scene(row,catalog,workcell):
    rng=np.random.default_rng(row['placement_seed'])
    x,y=[float(rng.uniform(*bounds)) for bounds in workcell['placement_xy_ranges_m']]
    yaw=float(rng.uniform(*workcell['yaw_range_rad']))
    T=pose(x,y,yaw,workcell['initial_bottom_clearance_m'])
    instances=[dict(object_id=row['object_id'],instance_id='target',role='target',world_from_object=T.tolist())]
    fixtures=[copy.deepcopy(workcell['table'])]
    if row['condition']=='bin_walls':fixtures+=copy.deepcopy(workcell['bin_walls'])
    if row['condition']=='neighbours':
        for i,other in enumerate(workcell['neighbours']):
            matrix=pose(*other['xy_m'],other['yaw_rad'],workcell['initial_bottom_clearance_m'])
            instances.append(dict(object_id=other['object_id'],instance_id=f'distractor_{i}',role='distractor',world_from_object=matrix.tolist()))
    spec=dict(row,episode_id=row['scene_id'],split='formal',instances=instances,scene_assets=fixtures)
    target_bounds=transform(catalog[row['object_id']]['bounds_m'],T)
    region=workcell['target_region_m']
    assert np.all(target_bounds[0]>=region['min']) and np.all(target_bounds[1]<=region['max']),('target outside pickup zone',row['scene_id'],target_bounds)
    table=np.array(workcell['table']['extents_m'])[:2]/2
    for instance in instances:
        bounds=transform(catalog[instance['object_id']]['bounds_m'],np.array(instance['world_from_object']))
        assert np.all(bounds[0,:2]>-table) and np.all(bounds[1,:2]<table),('outside table',row['scene_id'])
        if instance['role']=='distractor':
            assert np.any(bounds[1]<target_bounds[0]) or np.any(target_bounds[1]<bounds[0]),('initial object overlap',row['scene_id'])
    for fixture in fixtures[1:]:
        lower=np.array(fixture['center_m'])-np.array(fixture['extents_m'])/2
        upper=np.array(fixture['center_m'])+np.array(fixture['extents_m'])/2
        assert np.any(upper<target_bounds[0]) or np.any(target_bounds[1]<lower),('initial fixture overlap',row['scene_id'])
    task=dict(scene_id=row['scene_id'],split='formal',object_id=row['object_id'],condition=row['condition'],
        target_roi_world_m=copy.deepcopy(region),known_table_height_m=0.,
        target_selection='Fixed common task region and RGB foreground; no pose/depth/mask truth')
    return spec,task
