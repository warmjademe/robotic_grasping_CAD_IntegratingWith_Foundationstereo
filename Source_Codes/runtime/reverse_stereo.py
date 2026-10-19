"""Right-referenced FoundationStereo with explicit terminal algorithm failures."""
import argparse
import importlib.util
import time
from pathlib import Path
import cv2
import numpy as np
from fusion_common import ROOT,PROJECT,read,write,sha,gpu_lock
from inference_failures import AlgorithmFailure,InvalidPrediction,algorithm_deadline,failure_record,TERMINAL_ALGORITHM_STATUSES


def predict_reverse(predictor,images,budget):
    mirrored=[np.ascontiguousarray(images[1][:,::-1]),np.ascontiguousarray(images[0][:,::-1])]
    with algorithm_deadline(budget):
        result=predictor(mirrored)
        if not isinstance(result,np.ndarray) or result.shape!=images[0].shape[:2]:
            raise InvalidPrediction('Reverse disparity shape differs from the input image')
        return np.ascontiguousarray(result[:,::-1])


def main(source,output):
    backend=source/'code/perception_backend.py'
    spec=importlib.util.spec_from_file_location('frozen_reverse_backend',backend)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    predictor,provenance=module.make_predictor('foundationstereo')
    model=dict(provenance,backend_sha256=sha(backend),code_sha256=sha(__file__))
    if (output/'model.json').exists():
        previous=read(output/'model.json')
        for key in ['checkpoint_sha256','backend_sha256','code_sha256']:assert previous[key]==model[key]
    else:write(output/'model.json',model)
    protocol=read(source/'protocols/protocol.json');budget=protocol['budgets']['perception_timeout_wall_seconds']
    scenes=read(source/'datasets/manifest.json')['scenes']
    for scene in scenes:
        sid=scene['scene_id'];inp=source/'datasets/inputs'/sid;out=output/sid
        hashes={name:sha(inp/name) for name in ['left.png','right.png','cameras.json']}
        if (out/'record.json').exists():
            previous=read(out/'record.json');assert previous['input_sha256']==hashes
            assert previous['prediction_sha256']==sha(out/'prediction.npz')
            assert previous['status'] in {'completed',*TERMINAL_ALGORITHM_STATUSES}
            continue
        started=time.perf_counter();failure=None
        images=[cv2.cvtColor(cv2.imread(str(inp/f'{side}.png')),cv2.COLOR_BGR2RGB) for side in ['left','right']]
        baseline=source/'results/perception/foundationstereo'/sid/'perception.json'
        if read(baseline)['status'] in TERMINAL_ALGORITHM_STATUSES:
            failure=dict(status='upstream_algorithm_failure',failure_stage='perception',terminal_algorithm_failure=True,
                upstream_record_sha256=sha(baseline),upstream_record=str(baseline),automatic_retry=False,counts_as_success=False)
            disparity=np.zeros(images[0].shape[:2],np.float32)
        else:
            try:disparity=predict_reverse(predictor,images,budget)
            except AlgorithmFailure as error:
                failure=failure_record(error,'reverse_stereo',started,budget)
                disparity=np.zeros(images[0].shape[:2],np.float32)
        cal=read(inp/'cameras.json');f=cal['cameras']['right']['K'][0][0]
        valid=np.isfinite(disparity)&(disparity>.1)
        depth=np.full(disparity.shape,np.nan,np.float32)
        depth[valid]=f*cal['baseline_m']/disparity[valid];valid&=(depth>.05)&(depth<3)
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'prediction.npz',disparity=disparity,depth=depth,valid=valid)
        record=dict(scene_id=sid,status='completed' if failure is None else failure['status'],input_sha256=hashes,
            prediction_sha256=sha(out/'prediction.npz'),seconds=time.perf_counter()-started,reference_camera='right',
            disparity_convention='u_left-u_right',transformation='mirror_swap_predict_unmirror',ground_truth_input=False,
            prediction_is_failure_placeholder=failure is not None,code_sha256=sha(__file__))
        if failure:
            assert not valid.any();write(out/'failure.json',failure)
            record.update(failure_stage=failure['failure_stage'],failure_record_sha256=sha(out/'failure.json'))
        write(out/'record.json',record);print('REVERSE',sid,record['status'],flush=True)
    write(output/'status.json',dict(status='completed',scenes=len(scenes)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with gpu_lock():main(a.source,a.output)
