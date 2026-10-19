"""Expose already generated fused clouds to the unchanged GraspGen interface."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from project_common import read,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    tasks=read(a.source/'inputs/manifest.json')['tasks']
    for task in tasks:
        if task['perception']!='cad_fused':continue
        folder=a.source/'results/perception/cad_fused'/task['episode_id']
        record=read(folder/'record.json')
        assert record['cloud_sha256']==sha(folder/'cloud.npz')
        with np.load(folder/'cloud.npz') as data:
            target=len(data['target_points_m']);scene=len(data['scene_points_m'])
        shutil.copy2(folder/'cloud.npz',folder/'grasp_input.npz')
        write(folder/'perception.json',dict(grasp_input_ready=min(target,scene)>128,
            target_points=target,scene_points=scene,cloud_sha256=sha(folder/'cloud.npz'),
            geometry_truth_used=False,object_pose_used=False,cad_input=True,
            fusion_record_sha256=sha(folder/'record.json'),status='completed',
            input_format='GraspGen direct array reader; cloud arrays are preserved'))
        print('FUSED_INPUT',task['episode_id'],target,flush=True)


if __name__=='__main__':main()
