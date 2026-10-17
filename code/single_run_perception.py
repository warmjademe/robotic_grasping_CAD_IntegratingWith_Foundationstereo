"""Run one frozen stereo backend against a single-run batch's RGB inputs.

The existing batch backend owns preprocessing, disparity inference and cloud
export. This wrapper extends the FoundationStereo-only runner without editing
its frozen source, and preserves completed output and interrupted attempts.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time


METHODS = (
    'stereobm', 'sgm', 'foundationstereo', 'psmnet', 'raft_stereo',
    'igev_stereo', 'gwcnet',
)
INPUT_FILES = ('left.png', 'right.png', 'cameras.json', 'task.json')


def main(source, method):
    source = Path(source).expanduser().resolve()
    if method not in METHODS:
        raise ValueError('Unsupported stereo method: ' + method)
    sys.path.insert(0, str(source / 'code'))
    common = importlib.import_module('common')
    backend = importlib.import_module('perception_backend')
    failures = importlib.import_module('inference_failures')
    if common.RUN.resolve() != source:
        raise RuntimeError('Batch common module resolved to a different run')
    for module in (common, backend):
        if Path(module.__file__).resolve().parent != source / 'code':
            raise RuntimeError('Backend must come from the requested batch snapshot')
    if method not in common.PROTOCOL['stereo_methods']:
        raise ValueError('Method is not authorized by the batch protocol')
    import cv2
    import numpy as np

    read = lambda path: json.loads(Path(path).read_text())
    write, sha = common.write, common.sha
    manifest = read(source / 'datasets/manifest.json')
    freeze = read(source / 'protocols/input_freeze.json')
    if freeze['protocol_sha256'] != sha(source / 'protocols/protocol.json'):
        raise RuntimeError('Input freeze does not match the batch protocol')
    outroot = source / 'results/perception' / method
    outroot.mkdir(parents=True, exist_ok=True)
    pending = []
    for scene in manifest['scenes']:
        sid = scene['scene_id']
        inp = source / 'datasets/inputs' / sid
        out = outroot / sid
        hashes = {name: sha(inp / name) for name in INPUT_FILES}
        if hashes != freeze['inputs'][sid]:
            raise RuntimeError('Input hashes changed for ' + sid)
        recordpath = out / 'perception.json'
        if recordpath.exists():
            previous = read(recordpath)
            if previous['status'] not in {'completed', *failures.TERMINAL_ALGORITHM_STATUSES}:
                raise RuntimeError('Unresolved prior perception result for ' + sid)
            if previous['method'] != method or previous['input_sha256'] != hashes:
                raise RuntimeError('Prior perception identity differs for ' + sid)
            for name, field in [('cloud.npz', 'cloud_sha256'),
                                ('prediction.npz', 'prediction_sha256')]:
                if previous[field] != sha(out / name):
                    raise RuntimeError('Prior perception artifact changed: ' + str(out / name))
            if previous.get('grasp_input_ready'):
                for name, field in [('grasp_input.npz', 'grasp_input_sha256'),
                                    ('grasp_input.json', 'grasp_input_metadata_sha256')]:
                    if field in previous and previous[field] != sha(out / name):
                        raise RuntimeError('Prior grasp input changed: ' + str(out / name))
            if previous.get('failure_record_sha256'):
                if previous['failure_record_sha256'] != sha(out / 'failure.json'):
                    raise RuntimeError('Prior failure evidence changed for ' + sid)
            continue
        if out.exists() and any(out.iterdir()):
            raise RuntimeError('Preserve incomplete perception attempt before explicit recovery: ' + str(out))
        pending.append((sid, inp, out, hashes))
    if not pending:
        print(json.dumps(dict(method=method, status='all_existing_results_verified',
                              scenes=len(manifest['scenes']))), flush=True)
        return

    with common.gpu_lock():
        started = time.perf_counter()
        predictor, provenance = backend.make_predictor(method)
        model = dict(provenance, method=method, seed=0,
                     load_seconds=time.perf_counter() - started,
                     wrapper_sha256=sha(__file__),
                     backend_sha256=sha(backend.__file__),
                     algorithm_failure_handler_sha256=sha(failures.__file__))
        modelpath = outroot / 'model.json'
        if modelpath.exists():
            previous = read(modelpath)
            for field in ('method', 'seed', 'checkpoint_sha256', 'wrapper_sha256',
                          'backend_sha256', 'algorithm_failure_handler_sha256'):
                if previous.get(field) != model.get(field):
                    raise RuntimeError('Perception model identity changed: ' + field)
        else:
            write(modelpath, model)
        np.random.seed(0)
        for sid, inp, out, hashes in pending:
            if (source / 'ops/user_stop.json').exists():
                raise RuntimeError('User stop marker is present')
            task = read(inp / 'task.json')
            calibration = read(inp / 'cameras.json')
            images = []
            for side in ('left', 'right'):
                bgr = cv2.imread(str(inp / (side + '.png')))
                if bgr is None:
                    raise RuntimeError('Cannot read frozen RGB input: ' + str(inp / (side + '.png')))
                images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if images[0].shape != images[1].shape:
                raise RuntimeError('Stereo image dimensions differ for ' + sid)
            out.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            budget = common.PROTOCOL['budgets']['perception_timeout_wall_seconds']
            failure = None
            try:
                with failures.algorithm_deadline(budget):
                    disparity = predictor(images)
                    if not isinstance(disparity, np.ndarray) or disparity.shape != images[0].shape[:2]:
                        raise failures.InvalidPrediction('Stereo disparity shape differs from the input image')
            except failures.AlgorithmFailure as error:
                failure = failures.failure_record(error, 'perception', started, budget)
                disparity = np.zeros(images[0].shape[:2], np.float32)
            inference_seconds = time.perf_counter() - started
            _, x = np.indices(disparity.shape)
            valid = np.isfinite(disparity) & (disparity > 0) & (x - disparity >= 0)
            depth = np.zeros_like(disparity)
            fx = calibration['cameras']['left']['K'][0][0]
            depth[valid] = fx * calibration['stereo_baseline_m'] / disparity[valid]
            valid &= (depth > .15) & (depth < 1.)
            counts = backend.export_cloud(out, depth, valid, images, calibration, task, hashes)
            np.savez_compressed(out / 'prediction.npz', disparity=disparity, depth=depth, valid=valid)
            record = dict(scene_id=sid, method=method,
                          status='completed' if failure is None else failure['status'],
                          input_sha256=hashes, inference_seconds=inference_seconds,
                          seconds=time.perf_counter() - started, geometry_truth_used=False,
                          object_pose_used=False, cad_input=False,
                          cloud_sha256=sha(out / 'cloud.npz'),
                          prediction_sha256=sha(out / 'prediction.npz'),
                          prediction_is_failure_placeholder=failure is not None, **counts)
            if record['grasp_input_ready']:
                record.update(grasp_input_sha256=sha(out / 'grasp_input.npz'),
                              grasp_input_metadata_sha256=sha(out / 'grasp_input.json'))
            if failure:
                if record['grasp_input_ready'] or record['target_points'] != 0:
                    raise RuntimeError('Failed disparity unexpectedly produced a usable cloud')
                write(out / 'failure.json', failure)
                record.update(failure_stage='perception',
                              failure_record_sha256=sha(out / 'failure.json'))
            write(out / 'perception.json', record)
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    args = parser.parse_args()
    main(args.source, args.method)
