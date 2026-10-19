"""Run the original shared backends on new prediction-only clouds."""
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
from inference_failures import AlgorithmFailure,InvalidPrediction,algorithm_deadline,failure_record
from common import RUN, PROJECT, RUNTIME, GRASP_MODULES, PROTOCOL, sha, write, gpu_lock


def main(perception,grasp,limit=None):
    tasks=json.loads((RUN/'inputs/manifest.json').read_text())['tasks']
    tasks=[t for t in tasks if t['perception']==perception and t['grasp']==grasp]
    if limit is not None: tasks=tasks[:limit]
    cached=None
    if grasp=='graspgen':
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        sys.path.insert(0,str(RUNTIME/'upstream/GraspGen'));sys.path.insert(0,str(RUNTIME/'ops'))
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        from grasp_gen.grasp_server import GraspGenSampler,load_grasp_cfg
        from run_inference import seed_all,validate
        import graspgen_unique_voxels as adapter
        adapter.install()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        seed_all(0)
        torch.set_num_threads(8)
        cfg=load_grasp_cfg(str(RUNTIME/'models/checkpoints/graspgen_franka_panda.yml'))
        cached=GraspGenSampler(cfg)
        model_record=dict(checkpoint_config_sha256=sha(RUNTIME/'models/checkpoints/graspgen_franka_panda.yml'),
            checkpoint_sha256={str(p):sha(p) for p in [cfg.eval.checkpoint,cfg.discriminator.checkpoint]},model_initialization_seed=0,candidates=400,input_adapter=adapter.ADAPTER_VERSION,
            input_adapter_sha256=sha(adapter.__file__),deterministic_algorithms=True)
    for t in tasks:
        dest=RUN/'results/grasps'/t['task_id'];recordpath=dest/'task_result.json'
        if recordpath.exists(): continue
        cloud_dir=RUN/'results/perception'/perception/t['episode_id']
        status=json.loads((cloud_dir/'perception.json').read_text())
        if not status['grasp_input_ready']:
            write(recordpath,dict(task_id=t['task_id'],seed=int(t['seed']),status=status.get('upstream_failure_status','insufficient_points'),returned=0,input_sha256=status['cloud_sha256'],failure_stage=status.get('failure_stage','perception'),upstream_failure_record_sha256=status.get('upstream_failure_record_sha256')))
            continue
        start=time.perf_counter()
        if grasp=='graspgen':
            dest.mkdir(parents=True,exist_ok=True)
            with np.load(cloud_dir/'grasp_input.npz') as d:points=d['target_points_m'].astype(np.float32)
            poses=scores=None
            try:
                with algorithm_deadline(PROTOCOL['budgets']['grasp_timeout_wall_seconds']):
                    try:
                        adapter.RECORDS.clear()
                        seed_all(int(t['seed']))
                        poses,scores=GraspGenSampler.run_inference(points,cached,grasp_threshold=-1,num_grasps=400,
                            topk_num_grasps=400,min_grasps=400,max_tries=1,remove_outliers=False)
                        torch.cuda.synchronize();poses=poses.cpu().numpy();scores=scores.cpu().numpy();validate(poses,scores,400)
                        assert poses.shape==(400,4,4) and np.isfinite(poses).all() and np.isfinite(scores).all()
                        assert np.allclose(poses[:,3],[0,0,0,1],atol=1e-7)
                        assert np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-3)
                        assert np.allclose(poses[:,:3,:3].transpose(0,2,1)@poses[:,:3,:3],np.eye(3),atol=1e-3)
                    except (AssertionError,ValueError) as error:
                        raise InvalidPrediction('Invalid candidate poses or scores: '+str(error)) from error
            except AlgorithmFailure as error:
                failure=failure_record(error,'candidate_generation',start,PROTOCOL['budgets']['grasp_timeout_wall_seconds'])
                failure.update(task_id=t['task_id'],input_sha256=status['cloud_sha256'],source_sha256=sha(__file__))
                if isinstance(poses,np.ndarray) and isinstance(scores,np.ndarray):
                    np.savez_compressed(dest/'failed_prediction.npz',poses=poses,scores=scores)
                    failure['raw_prediction_sha256']=sha(dest/'failed_prediction.npz')
                write(dest/'failure.json',failure)
                write(recordpath,dict(task_id=t['task_id'],seed=int(t['seed']),status=error.status,returned=0,
                    input_sha256=status['cloud_sha256'],failure_stage='candidate_generation',failure_record_sha256=sha(dest/'failure.json')))
                continue
            A=np.array([[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
            widths=np.full(len(scores),.08)
            np.savez_compressed(dest/'grasps.npz',T_world_grasp=poses@A,score=scores,jaw_width_m=widths,
                gripper_width_valid=np.ones(len(scores),dtype=bool),model_poses=poses,T_model_grasp=A)
            write(dest/'grasp_generation.json',dict({**model_record,'seed':int(t['seed'])},input_npz_sha256=sha(cloud_dir/'grasp_input.npz'),
                output_sha256=sha(dest/'grasps.npz'),returned=len(scores),seconds=time.perf_counter()-start,
                source_sha256=sha(__file__),physical_robot_used=False,sparse_input_conversion=adapter.RECORDS))
        else:
            log=RUN/'logs'/f"grasp_{t['task_id']}.log"
            cmd=[str(RUNTIME/'.venv-infer/bin/python'),str(GRASP_MODULES/'shared/infer.py'),
                '--method',grasp,'--input',str(cloud_dir/'grasp_input.npz'),'--out',str(dest),'--seed',str(int(t['seed']))]
            with log.open('w') as f:
                try:
                    result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=PROTOCOL['budgets']['grasp_timeout_wall_seconds'])
                    result.check_returncode()
                except BaseException as error:
                    write(dest/'failure.json',dict(status='backend_error',error=repr(error),task_id=t['task_id'],log=str(log.relative_to(RUN))))
                    raise
        info=json.loads((dest/'grasp_generation.json').read_text())
        record=dict(task_id=t['task_id'],seed=int(t['seed']),status='completed' if info['returned'] else 'no_candidates',
            returned=info['returned'],seconds=time.perf_counter()-start,grasps_sha256=sha(dest/'grasps.npz'),
            input_sha256=sha(cloud_dir/'cloud.npz'))
        write(recordpath,record);print(json.dumps(record),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--perception',required=True);p.add_argument('--grasp',required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    with gpu_lock(): main(a.perception,a.grasp,a.limit)
