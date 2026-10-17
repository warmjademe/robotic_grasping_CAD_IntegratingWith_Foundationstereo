"""Ground truth is read here only to instantiate evaluation physics scenes."""
import json
import numpy as np
import trimesh
from common import RUN, STUDY, translation
from franka_kinematics import Panda, HOME, DATASET_TO_ROBOT


def calibration():
    H=np.eye(4); H[:3,:3]=np.diag([1,-1,-1]); H[:3,3]=[.5,0,.1165]
    robot=Panda(); pre=H.copy(); pre[2,3]+=.12; lift=H.copy(); lift[2,3]+=.15
    qpre,error=robot.solve(pre,HOME)
    assert error[0] < .0002 and error[1] < np.deg2rad(.1), error
    paths=dict(transit=np.linspace(HOME,qpre,81).tolist()); seed=qpre
    for phase,A,B in [('approach',pre,H),('lift',H,lift)]:
        paths[phase]=[]
        for f in np.linspace(0,1,61):
            T=A.copy(); T[:3,3]=(1-f)*A[:3,3]+f*B[:3,3]
            seed,error=robot.solve(T,seed)
            assert error[0] < .0002 and error[1] < np.deg2rad(.1), error
            paths[phase].append(seed.tolist())
    mesh=RUN/'assets/calibration_cube.obj'
    if not mesh.exists(): trimesh.creation.box([.03]*3).export(mesh)
    cases=[]
    for name,dx,opened in [('centered_box',0.,False),('far_negative',.2,False),('open_negative',0.,True),('collision_blocked',.2,False)]:
        fixtures=[dict(name='table',extents_m=[.65,.5,.035],center_m=[0,0,-.0175])]
        if name=='collision_blocked': fixtures.append(dict(name='obstacle',extents_m=[.15,.15,.10],center_m=[0,0,.08]))
        scene=dict(scene_assets=fixtures,instances=[dict(object_id='calibration_cube',role='target',instance_id='target',world_from_object=translation([dx,0,.015]).tolist())])
        cases.append(dict(task_id=name,expected=name=='centered_box',scene=scene,mesh=str(mesh),
            selection=dict(paths=paths,hand_pose=H.tolist(),opening=.08,closing=.08 if opened else 0.)))
    return cases


def cases(task_ids):
    if task_ids==['calibration']: return calibration()
    manifest=json.loads((RUN/'inputs/manifest.json').read_text())
    lookup={t['task_id']:t for t in manifest['tasks']}
    result=[]
    for task_id in task_ids:
        task=lookup[task_id]
        plan=json.loads((RUN/'results/plans'/f'{task_id}.json').read_text())
        assert plan['status']=='sampled_path_found'
        scene=STUDY/'datasets/evaluation'/task['episode_id']/'scene.json'
        result.append(dict(**task,scene=json.loads(scene.read_text()),
            mesh=str(STUDY/'datasets/assets'/task['object_id']/'mesh_m.ply'),selection=plan['selected']))
    return result
