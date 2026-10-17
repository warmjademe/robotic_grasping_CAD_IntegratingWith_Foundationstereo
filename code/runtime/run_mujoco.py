"""Execute shared Panda plans in MuJoCo with free physical workpieces."""
import argparse
import importlib.metadata
import json
import socket
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh
import vhacdx
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from common import RUN, STUDY, PROTOCOL, command, outcome, pose7, sha, write
from franka_kinematics import DATASET_TO_ROBOT, Panda
from scenes import cases

ASSET=RUN/'assets/menagerie/franka_emika_panda'
CFG=PROTOCOL['physics']


def text(values): return ' '.join(str(float(x)) for x in values)


def convex_parts(mesh_path):
    key=sha(mesh_path); cache=RUN/'assets/mujoco_collision_v2'/key
    marker=cache/'parts.json'
    if marker.exists(): return [cache/p for p in json.loads(marker.read_text())['files']]
    cache.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load(mesh_path,force='mesh')
    if mesh.is_convex: parts=[mesh.convex_hull]
    else:
        faces=np.column_stack([np.full(len(mesh.faces),3),mesh.faces]).ravel().astype(np.uint32)
        raw=vhacdx.compute_vhacd(np.asarray(mesh.vertices,dtype=np.float64),faces,
            maxConvexHulls=64,resolution=400000,asyncACD=False)
        parts=[]; repaired=[]
        for i,(vertices,faces) in enumerate(raw):
            part=trimesh.Trimesh(vertices,faces,process=True).convex_hull
            if not part.is_volume:
                # Trimesh 4.5.3 can discard a near-degenerate hull triangle,
                # leaving a hole. Retain all Qhull simplices of the same points;
                # orient them from Qhull's outward plane normals. No dilation
                # or workpiece-wide convexification is applied.
                vertices=np.asarray(vertices)
                center=vertices.mean(axis=0);scale=np.ptp(vertices,axis=0).max()
                normalized=(vertices-center)/scale
                hull=ConvexHull(normalized)
                triangles=hull.simplices.copy();xyz=normalized[triangles]
                normals=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
                reverse=np.einsum('ij,ij->i',normals,hull.equations[:,:3])<0
                triangles[reverse]=triangles[reverse][:,::-1]
                part=trimesh.Trimesh(vertices,triangles,process=False)
                assert np.isclose(part.volume,hull.volume*scale**3,rtol=1e-8,atol=1e-15)
                repaired.append(i)
            parts.append(part)
    assert parts and all(p.is_volume and p.volume>0 for p in parts)
    files=[]
    for i,part in enumerate(parts):
        file=f'part_{i:03d}.obj'; part.export(cache/file); files.append(file)
    write(marker,dict(files=files,source_sha256=key,source=str(mesh_path),vhacdx=importlib.metadata.version('vhacdx'),
        maxConvexHulls=64,resolution=400000,asyncACD=False,
        topology_repaired_parts=[] if mesh.is_convex else repaired,
        topology_repair='Retain and orient all Qhull simplices; unchanged hull vertices and volume'))
    return [cache/f for f in files]


def build(case,friction,out):
    root=ET.parse(ASSET/'panda.xml').getroot()
    root.find('compiler').set('meshdir',str(ASSET/'assets'))
    option=root.find('option')
    option.attrib.update(timestep=str(1/CFG['hz']),gravity='0 0 -9.81',integrator='implicitfast',
        solver='Newton',iterations='100',tolerance='1e-10',cone='elliptic')
    for tag in ['actuator','tendon','equality','keyframe']:
        for el in root.findall(tag):root.remove(el)
    robot_root=root.find('worldbody')
    for body in robot_root.iter('body'):body.set('gravcomp','1')
    for j in root.iter('joint'):
        if 'type' not in j.attrib or j.get('type') in ['hinge','slide']:
            j.attrib.update(damping='0',armature='0')
    act=ET.SubElement(root,'actuator')
    for i,limit in enumerate(CFG['arm_effort_limits_nm'],1):
        ET.SubElement(act,'position',name=f'arm{i}',joint=f'joint{i}',kp=str(CFG['arm_kp']),kv=str(CFG['arm_kd']),
            forcelimited='true',forcerange=f'-{limit} {limit}',ctrllimited='false')
    for i in [1,2]:
        ET.SubElement(act,'position',name=f'finger{i}',joint=f'finger_joint{i}',kp=str(CFG['finger_kp']),kv=str(CFG['finger_kd']),
            forcelimited='true',forcerange='-20 20',ctrllimited='true',ctrlrange='0 .04')
    scene=case['scene']; asset=root.find('asset')
    for fixture in scene['scene_assets']:
        xyz=DATASET_TO_ROBOT[:3,3]+np.array(fixture['center_m'])
        ET.SubElement(robot_root,'geom',name='fixture_'+fixture['name'],type='box',size=text(np.array(fixture['extents_m'])/2),pos=text(xyz),rgba='.4 .4 .4 1')
    parts_info=[]
    for i,inst in enumerate(scene['instances']):
        target=inst['role']=='target'
        mesh=Path(case['mesh']) if target else STUDY/'datasets/assets'/inst['object_id']/'mesh_m.ply'
        T=DATASET_TO_ROBOT@np.array(inst['world_from_object']); p=pose7(T)
        name='target' if target else f'distractor_{i}'
        body=ET.SubElement(robot_root,'body',name=name,pos=text(p[:3]),quat=text(p[3:]))
        ET.SubElement(body,'freejoint',name=name+'_free')
        paths=convex_parts(mesh); hulls=[trimesh.load(p,force='mesh') for p in paths]
        weights=np.array([abs(m.volume) for m in hulls]);weights/=weights.sum()
        for j,(path,weight) in enumerate(zip(paths,weights)):
            mn=f'{name}_mesh_{j}'
            ET.SubElement(asset,'mesh',name=mn,file=str(path))
            ET.SubElement(body,'geom',name=f'{name}_shape_{j}',type='mesh',mesh=mn,mass=str(CFG['object_mass_kg']*weight),rgba='.3 .6 .8 1')
        parts_info.append(dict(object_id=inst['object_id'],role=inst['role'],convex_parts=len(paths),mesh_sha256=sha(mesh)))
    for geom in root.iter('geom'):
        geom.attrib.update(friction=f'{friction} .005 .0001',condim='3',solref='.008 1',solimp='.9 .95 .001',margin='0')
    ET.ElementTree(root).write(out/'scene.xml')
    model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
    data=mujoco.MjData(model)
    return model,data,parts_info


def execute(case,profile,dest):
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists(): raise FileExistsError(dest)
    started=time.perf_counter()
    friction=CFG['profiles'][profile]
    model,data,parts=build(case,friction,dest)
    compile_seconds=time.perf_counter()-started
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in [f'joint{i}' for i in range(1,8)]+['finger_joint1','finger_joint2']]
    addresses=model.jnt_qposadr[ids]
    target=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'target')
    hand=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'hand')
    fingers={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ['left_finger','right_finger']}
    arm={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in [f'link{i}' for i in range(1,8)]+['hand']}
    selection=case['selection']; q0=np.array(selection['paths']['transit'][0])
    data.qpos[addresses]=np.r_[q0,[selection['opening']/2]*2]
    data.ctrl[:]=data.qpos[addresses];mujoco.mj_forward(model,data)
    fk=Panda().fk(q0)
    fk_error=float(np.linalg.norm(fk[:3,3]-data.xpos[hand]))
    fk_rotation=float((Rotation.from_matrix(fk[:3,:3]).inv()*Rotation.from_matrix(data.xmat[hand].reshape(3,3))).magnitude())
    assert fk_error<1e-6 and fk_rotation<1e-6,(fk_error,fk_rotation)
    frames=[]; phases=[]; times=[]; qs=[]; forces=[]
    bad_steps=0;max_arm=0.;max_finger=0.;step=0
    endpoint=None;baseline=None
    simulated_at=time.perf_counter()
    for phase,duration in CFG['phases_seconds']:
        n=round(duration*CFG['hz'])
        for j in range(n):
            q,opening=command(selection,phase,(j+1)/n)
            data.ctrl[:]=np.r_[q,[opening/2]*2]
            mujoco.mj_step(model,data)
            if not np.all(np.isfinite(data.qpos)):raise RuntimeError('Nonfinite physical state')
            # Match Isaac's resultant normal force gate, retaining all contact
            # forces in additional per-frame diagnostics would use a different gate.
            net=np.zeros((model.nbody,3)); ft=np.zeros((model.nbody,3))
            for k in range(data.ncon):
                contact=data.contact[k]; a,b=model.geom_bodyid[contact.geom]
                force=np.zeros(6);mujoco.mj_contactForce(model,data,k,force)
                world=contact.frame.reshape(3,3)[0]*force[0]
                net[a]-=world;net[b]+=world
                if a in fingers and b==target:ft[a]-=world
                if b in fingers and a==target:ft[b]+=world
            af=float(np.linalg.norm(net[list(arm)],axis=1).max())
            ff=float(np.linalg.norm((net-ft)[list(fingers)],axis=1).max())
            if phase!='settle':
                bad_steps+=int(af>.5 or ff>.5); max_arm=max(max_arm,af);max_finger=max(max_finger,ff)
            step+=1
            if step%(CFG['hz']//CFG['trace_hz'])==0:
                frames.append(np.r_[data.xpos[target].copy(),data.xquat[target].copy()])
                qs.append(data.qpos[addresses].copy());phases.append(phase);times.append(data.time);forces.append([af,ff])
        if phase=='settle':baseline=data.xpos[target].copy()
        if phase=='approach_hold':
            H=np.array(selection['hand_pose'])
            endpoint=(float(np.linalg.norm(data.xpos[hand]-H[:3,3])),float(np.rad2deg((Rotation.from_matrix(data.xmat[hand].reshape(3,3)).inv()*Rotation.from_matrix(H[:3,:3])).magnitude())))
    simulation_wall=time.perf_counter()-simulated_at
    mesh=trimesh.load(case['mesh'],force='mesh')
    row=outcome(frames,phases,mesh.vertices,baseline,endpoint,bad_steps,max_arm,max_finger)
    row.update(task_id=case['task_id'],expected=case.get('expected'),platform='franka_mujoco',profile=profile)
    np.savez_compressed(dest/'trace.npz',time_s=times,phases=phases,object_pose=frames,joints=qs,contact_resultants=forces)
    report=dict(status='completed',rows=[row],host=socket.gethostname(),platform='franka_mujoco',profile=profile,
        seconds=time.perf_counter()-started,compile_seconds=compile_seconds,simulation_wall_seconds=simulation_wall,
        simulated_seconds=float(data.time),parallelism=1,object_collision_parts=parts,
        fk_position_error_m=fk_error,fk_rotation_error_rad=fk_rotation,protocol_sha256=sha(RUN/'protocols/protocol.json'),
        code_sha256=sha(__file__),versions={k:importlib.metadata.version(k) for k in ['mujoco','numpy','scipy','trimesh','vhacdx']},
        robot_asset_commit='d7b481bac63d3d600ab404ee74f7474c5efe8a67',
        robot_asset_sha256=sha(ASSET/'panda.xml'),object_attached=False,
        notes='Menagerie native collision and inertia; robot gravity compensation, independent finger PD; no scene-pose feedback control.')
    write(dest/'result.json',report)
    print(json.dumps(row),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',nargs='+',required=True)
    p.add_argument('--profile',choices=list(CFG['profiles']),default='nominal')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for case in cases(a.tasks):execute(case,a.profile,a.output/case['task_id'])
