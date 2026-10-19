"""Prediction-only Panda kinematics and collision checking from archived URDF.

Robot collision assets are the archived GraspGen Panda meshes; physics assets
are independently identified. No object identity, CAD or true pose is read.
"""
from pathlib import Path
import os
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from common import PROTOCOL, translation

BASE = Path(os.environ['GRASPGEN_RUNTIME']).resolve()
URDF = BASE / 'upstream/GraspGen/assets/franka/franka_panda.urdf'
DATASET_TO_ROBOT = translation(PROTOCOL['planning']['dataset_to_robot_translation_m'])
HOME = np.array(PROTOCOL['planning']['home_joints_rad'])

def origin(node):
    T = np.eye(4)
    if node is not None:
        T[:3,3] = np.fromstring(node.get('xyz','0 0 0'), sep=' ')
        T[:3,:3] = Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return T

class Panda:
    def __init__(self):
        self.root = ET.parse(URDF).getroot()
        self.joints = [self.root.find(f"joint[@name='panda_joint{i}']") for i in range(1,8)]
        self.origins = [origin(j.find('origin')) for j in self.joints]
        self.lower = np.array([float(j.find('limit').get('lower')) for j in self.joints])
        self.upper = np.array([float(j.find('limit').get('upper')) for j in self.joints])
        self.tail = origin(self.root.find("joint[@name='panda_joint8']/origin")) @ origin(self.root.find("joint[@name='panda_hand_joint']/origin"))

    def frames(self, q, opening=.08, jacobian=False):
        T = np.eye(4); frames = {'panda_link0': T.copy()}; axes=[]; positions=[]
        for i,(O,a) in enumerate(zip(self.origins,q),1):
            T = T @ O
            axes.append(T[:3,2].copy());positions.append(T[:3,3].copy())
            c,s = np.cos(a),np.sin(a)
            Z = np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
            T = T @ Z; frames[f'panda_link{i}'] = T.copy()
        T = T @ self.tail; frames['panda_hand'] = T
        for name,sign in [('panda_leftfinger',1),('panda_rightfinger',-1)]:
            j = next(j for j in self.root.findall('joint') if j.find('child').get('link')==name)
            frames[name] = T @ origin(j.find('origin')) @ translation([0, sign*opening/2, 0])
        if jacobian:
            J = np.vstack([np.array([np.cross(a,T[:3,3]-p) for a,p in zip(axes,positions)]).T,np.array(axes).T])
            return T,J
        return frames

    def fk(self,q): return self.frames(q)['panda_hand']

    def solve(self, desired, seed, max_iterations=100):
        q=np.clip(np.asarray(seed).copy(),self.lower+1e-6,self.upper-1e-6)
        best=(float('inf'),q.copy(),(float('inf'),float('inf')))
        for _ in range(max_iterations):
            T,J=self.frames(q,jacobian=True)
            dp=desired[:3,3]-T[:3,3]
            dr=Rotation.from_matrix(desired[:3,:3]@T[:3,:3].T).as_rotvec()
            errors=(np.linalg.norm(dp),np.linalg.norm(dr))
            loss=errors[0]+.2*errors[1]
            if loss<best[0]:best=(loss,q.copy(),errors)
            if errors[0]<.00015 and errors[1]<np.deg2rad(.05):return q,errors
            J[3:]*=.2; e=np.r_[dp,.2*dr]
            dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-5,e)
            dq*=min(1.,.2/max(np.max(np.abs(dq)),1e-9))
            q=np.clip(q+dq,self.lower+1e-6,self.upper-1e-6)
        return best[1],best[2]

def fixtures(condition):
    workcell=PROTOCOL['workcell']
    values=[workcell['table']]
    if condition=='bin_walls':values+=workcell['bin_walls']
    elif condition not in ['single','neighbours']:raise ValueError(condition)
    return [(v['name'],v['extents_m'],v['center_m']) for v in values]

class Collision:
    def __init__(self, robot, condition, scene):
        self.robot=robot;self.cm=trimesh.collision.CollisionManager();self.env=trimesh.collision.CollisionManager()
        self.names=[];self.transforms={}
        for link in robot.root.findall('link'):
            for i,c in enumerate(link.findall('collision')):
                mesh=c.find('geometry/mesh')
                if mesh is None:continue
                name=link.get('name')
                geom=trimesh.load(URDF.parent/mesh.get('filename'),force='mesh')
                geom.apply_scale(np.fromstring(mesh.get('scale','1 1 1'),sep=' '))
                self.cm.add_object(name,geom);self.names.append(name);self.transforms[name]=origin(c.find('origin'))
        self.allowed=set()
        chain=[f'panda_link{i}' for i in range(8)]+['panda_hand']
        self.allowed.update(frozenset([a,b]) for a,b in zip(chain,chain[1:]))
        self.allowed.update([frozenset(['panda_link6','panda_hand']),frozenset(['panda_hand','panda_leftfinger']),frozenset(['panda_hand','panda_rightfinger'])])
        for n,size,pos in fixtures(condition):self.env.add_object(n,trimesh.creation.box(size),DATASET_TO_ROBOT@translation(pos))
        keep=(scene[:,2]>.003)&(scene[:,2]<.30)&(np.abs(scene[:,0])<.325)&(np.abs(scene[:,1])<.25)
        points=scene[keep];self.voxels=0
        if len(points):
            voxel=PROTOCOL['planning']['scene_voxel_m']
            cells=np.unique(np.floor(points/voxel).astype(np.int64),axis=0);self.voxels=len(cells)
            self.env.add_object('perceived_scene',trimesh.voxel.ops.multibox((cells+.5)*voxel,pitch=voxel),DATASET_TO_ROBOT)

    def check(self,q,opening):
        frames=self.robot.frames(q,opening)
        for name in self.names:self.cm.set_transform(name,frames[name]@self.transforms[name])
        _,inside=self.cm.in_collision_internal(return_names=True)
        inside=[sorted(pair) for pair in inside if frozenset(pair) not in self.allowed]
        _,outside=self.cm.in_collision_other(self.env,return_names=True)
        return dict(collision=bool(inside or outside),self_pairs=sorted(inside),environment_pairs=sorted([list(p) for p in outside]),perceived_voxels=self.voxels)
