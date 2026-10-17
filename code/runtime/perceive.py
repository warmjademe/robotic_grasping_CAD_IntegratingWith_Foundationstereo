"""Frozen FoundationStereo inference retaining timed-out and empty tasks."""
import argparse
import json
import time
import numpy as np
import cv2
from common import RUN,PROTOCOL,write,sha,gpu_lock
from perception_backend import make_predictor,export_cloud
from inference_failures import AlgorithmFailure,InvalidPrediction,algorithm_deadline,failure_record,TERMINAL_ALGORITHM_STATUSES


def predict_or_failure(predictor,images,budget):
    started=time.perf_counter()
    try:
        with algorithm_deadline(budget):
            disparity=predictor(images)
            if not isinstance(disparity,np.ndarray) or disparity.shape!=images[0].shape[:2]:
                raise InvalidPrediction('Stereo disparity shape differs from the input image')
    except AlgorithmFailure as error:
        return np.zeros(images[0].shape[:2],np.float32),failure_record(error,'perception',started,budget)
    return disparity,None


def main(method):
    assert method=='foundationstereo'
    manifest=json.loads((RUN/'datasets/manifest.json').read_text())
    freeze=json.loads((RUN/'protocols/input_freeze.json').read_text())
    assert freeze['protocol_sha256']==sha(RUN/'protocols/protocol.json')
    outroot=RUN/'results/perception'/method;outroot.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();predictor,provenance=make_predictor(method)
    model=dict(provenance,load_seconds=time.perf_counter()-start,seed=0,
        wrapper_sha256=sha(__file__),backend_sha256=sha(RUN/'code/perception_backend.py'))
    modelpath=outroot/'model.json'
    if modelpath.exists():
        previous=json.loads(modelpath.read_text())
        for field in ['checkpoint_sha256','wrapper_sha256','backend_sha256']:assert previous[field]==model[field]
    else:write(modelpath,model)
    np.random.seed(0)
    for scene in manifest['scenes']:
        sid=scene['scene_id'];out=outroot/sid;inp=RUN/'datasets/inputs'/sid
        hashes={name:sha(inp/name) for name in ['left.png','right.png','cameras.json','task.json']}
        assert hashes==freeze['inputs'][sid]
        if (out/'perception.json').exists():
            previous=json.loads((out/'perception.json').read_text())
            assert previous['status'] in {'completed',*TERMINAL_ALGORITHM_STATUSES}
            assert previous['input_sha256']==hashes and previous['cloud_sha256']==sha(out/'cloud.npz')
            assert previous['prediction_sha256']==sha(out/'prediction.npz')
            continue
        out.mkdir(parents=True,exist_ok=True)
        task=json.loads((inp/'task.json').read_text());cal=json.loads((inp/'cameras.json').read_text())
        images=[cv2.cvtColor(cv2.imread(str(inp/(side+'.png'))),cv2.COLOR_BGR2RGB) for side in ['left','right']]
        started=time.perf_counter();disparity,failure=predict_or_failure(predictor,images,PROTOCOL['budgets']['perception_timeout_wall_seconds'])
        inference=time.perf_counter()-started
        _,x=np.indices(disparity.shape)
        valid=np.isfinite(disparity)&(disparity>0)&(x-disparity>=0)
        depth=np.zeros_like(disparity);depth[valid]=cal['cameras']['left']['K'][0][0]*cal['stereo_baseline_m']/disparity[valid]
        valid&=(depth>.15)&(depth<1.)
        counts=export_cloud(out,depth,valid,images,cal,task,hashes)
        np.savez_compressed(out/'prediction.npz',disparity=disparity,depth=depth,valid=valid)
        record=dict(scene_id=sid,method=method,status='completed' if failure is None else failure['status'],input_sha256=hashes,
            inference_seconds=inference,seconds=time.perf_counter()-started,geometry_truth_used=False,object_pose_used=False,cad_input=False,
            cloud_sha256=sha(out/'cloud.npz'),prediction_sha256=sha(out/'prediction.npz'),prediction_is_failure_placeholder=failure is not None,**counts)
        if failure:
            assert not record['grasp_input_ready'] and record['target_points']==0
            write(out/'failure.json',failure);record.update(failure_stage='perception',failure_record_sha256=sha(out/'failure.json'))
        write(out/'perception.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['foundationstereo'],required=True);a=p.parse_args()
    with gpu_lock():main(a.method)
