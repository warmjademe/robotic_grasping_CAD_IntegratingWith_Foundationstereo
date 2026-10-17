"""Render calibrated stereo observations after physical settling in Isaac Sim.

Only RGB/calibration/task metadata go to inputs/. True depth and object poses
go to evaluation/. Inference processes never read scene_specs or evaluation.
"""
import os
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
import json
import sys
import time
from pathlib import Path
from common import RUN, RUNTIME, PROTOCOL, gpu_lock, pose7, write, sha
lock = gpu_lock(); lock.__enter__()
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
import isaaclab.sim as su
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from pxr import UsdGeom, UsdShade
import omni.usd
sys.path.insert(0,str(RUNTIME/'upstream/GraspDataGen/scripts/graspgen'))
from mesh_utils import convert_mesh_to_usd

manifest=json.loads((RUN/'datasets/manifest.json').read_text())
specs=[json.loads((RUN/'datasets/scene_specs'/f"{s['scene_id']}.json").read_text()) for s in manifest['scenes']]
catalog={x['object_id']:x for x in json.loads((RUN/'datasets/object_catalog.json').read_text())}
sim=su.SimulationContext(su.SimulationCfg(dt=1/250,device='cuda:0'))
stage=omni.usd.get_context().get_stage()
origins=np.array([[i%6*2.,i//6*2.,0.] for i in range(len(specs))])
material=su.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.)
look_cfg=su.PreviewSurfaceCfg(diffuse_color=(.55,.58,.61),roughness=.35,metallic=.15)
look_cfg.func('/World/Looks/Workpiece',look_cfg)
workpiece_material=UsdShade.Material.Get(stage,'/World/Looks/Workpiece')
usd={}
for key,obj in catalog.items():
    dest=RUN/'assets'/f'{key}_render.usd';dest.parent.mkdir(parents=True,exist_ok=True)
    convert_mesh_to_usd(str(dest),str(RUN/obj['mesh']),mass=.1,collision_approximation='convexDecomposition',
        physics_material=material)
    usd[key]=str(dest)
objects=[]
for i,spec in enumerate(specs):
    prefix=f'/World/envs/env_{i}'
    UsdGeom.Xform.Define(stage,prefix)
    for fixture in spec['scene_assets']:
        table=su.CuboidCfg(size=tuple(fixture['extents_m']),collision_props=su.CollisionPropertiesCfg(),
            physics_material=material,visual_material=su.PreviewSurfaceCfg(diffuse_color=(.18,.23,.30),roughness=.8))
        table.func(prefix+'/'+fixture['name'],table,translation=tuple(origins[i]+fixture['center_m']))
    group=[]
    for inst in spec['instances']:
        T=np.array(inst['world_from_object']);T[:3,3]+=origins[i];p=pose7(T)
        obj=RigidObject(RigidObjectCfg(prim_path=prefix+'/'+inst['instance_id'],
            spawn=su.UsdFileCfg(usd_path=usd[inst['object_id']],
                rigid_props=su.RigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=su.CollisionPropertiesCfg(contact_offset=.0005,rest_offset=0.)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(p[:3]),rot=tuple(p[3:]))))
        group.append(obj)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prefix+'/'+inst['instance_id'])).Bind(workpiece_material)
    objects.append(group)
light=su.DomeLightCfg(intensity=1800.,color=(.9,.93,1.));light.func('/World/Light',light)
camera_cfg=PROTOCOL['camera'];w,h=camera_cfg['width'],camera_cfg['height'];f=camera_cfg['focal_px']
cameras=[]
for side in ['left','right']:
    cameras.append(Camera(CameraCfg(prim_path='/World/'+side,height=h,width=w,
        data_types=['rgb','distance_to_image_plane'],
        spawn=su.PinholeCameraCfg(focal_length=f/w*20.955,horizontal_aperture=20.955,
                                 clipping_range=(.01,10.)))))
sim.reset()
for _ in range(375):
    sim.step(render=False)
    for group in objects:
        for obj in group:obj.update(1/250)
eye=np.array(camera_cfg['center_eye_m']);look=np.array(camera_cfg['look_at_m'])
z=(look-eye);z/=np.linalg.norm(z);x=np.cross(z,[0,0,1]);x/=np.linalg.norm(x);y=np.cross(z,x)
R=np.column_stack([x,y,z]);q=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
started=time.perf_counter()
for i,(spec,group) in enumerate(zip(specs,objects)):
    scene_id=spec['scene_id'];inp=RUN/'datasets/inputs'/scene_id;ev=RUN/'datasets/evaluation'/scene_id
    ev.mkdir(parents=True,exist_ok=True)
    calibration={'width':w,'height':h,'baseline_m':camera_cfg['baseline_m'],'cameras':{}}
    for index,(side,camera) in enumerate(zip(['left','right'],cameras)):
        E=eye+x*camera_cfg['baseline_m']*(index-.5)
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=E
        camera.set_world_poses(torch.tensor([E+origins[i]],device='cuda',dtype=torch.float32),
            torch.tensor([q],device='cuda',dtype=torch.float32),convention='ros')
        for _ in range(10):sim.render()
        camera.update(1/30,force_recompute=True)
        rgb=camera.data.output['rgb'][0,...,:3].cpu().numpy()
        depth=camera.data.output['distance_to_image_plane'][0].cpu().numpy().squeeze()
        K=camera.data.intrinsic_matrices[0].cpu().numpy()
        assert np.isclose(K[0,0],f,rtol=1e-4),K
        Image.fromarray(rgb.astype(np.uint8)).save(inp/f'{side}.png')
        np.save(ev/f'{side}_depth_gt.npy',depth)
        calibration['cameras'][side]={'K':K.tolist(),'world_from_camera':T.tolist()}
    write(inp/'calibration.json',calibration)
    write(inp/'cameras.json',dict(calibration,stereo_baseline_m=camera_cfg['baseline_m']))
    for inst,obj in zip(spec['instances'],group):
        p=obj.data.root_state_w[0,:7].cpu().numpy();T=np.eye(4)
        T[:3,:3]=Rotation.from_quat(p[[4,5,6,3]]).as_matrix();T[:3,3]=p[:3]-origins[i]
        inst['world_from_object']=T.tolist()
    write(ev/'scene.json',spec)
    write(inp/'render_record.json',{'renderer':'Isaac Sim RTX','source_meshes':'T-LESS and ITODD manual CAD in recorded support frames',
        'physics_settling_s':1.5,'rgb_sha256':{side:sha(inp/f'{side}.png') for side in ['left','right']},
        'ground_truth_in_inputs':False,'camera_parallel_rectified':True})
    print('RENDERED',scene_id,flush=True)
write(RUN/'datasets/render_complete.json',dict(scene_count=len(specs),seconds=time.perf_counter()-started,
    renderer='Isaac Sim RTX',calibration='Parallel OpenCV optical frames; 0.08 m baseline; actual SDK intrinsics'))
app.close(wait_for_replicator=False,skip_cleanup=True)
