#!/usr/bin/env python3
"""Industrial RGB-only development inference. This process never opens evaluation/.

One method per process avoids author packages with conflicting module names.
Frozen official models are reused; target selection uses a declared workspace ROI.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import signal
from unittest.mock import patch

os.environ.setdefault('XFORMERS_DISABLED', '1')
PROJECT = Path(os.environ['CAD_RELEASE_ROOT']).resolve()
STEREO = Path(os.environ['STEREO_MODULE_ROOT']).resolve()
GRASP = Path(os.environ['GRASP_MODULE_ROOT']).resolve()
sys.path.insert(0, str(STEREO / 'baseline_runtime/python_deps'))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'python_deps'))
sys.path.insert(0, str(GRASP / 'shared'))
import cv2
import numpy as np
from scipy.spatial import cKDTree
from contract import save, sha

METHODS = ['stereobm', 'sgm', 'foundationstereo', 'psmnet', 'raft_stereo', 'igev_stereo', 'gwcnet']


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def make_predictor(method):
    if method in ['stereobm', 'sgm']:
        cv2.setNumThreads(8)
        if method == 'stereobm':
            model = cv2.StereoBM_create(numDisparities=192, blockSize=15)
        else:
            model = cv2.StereoSGBM_create(minDisparity=0, numDisparities=192,
                blockSize=5, P1=8*25, P2=32*25, disp12MaxDiff=1,
                uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        def predict(images):
            gray = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in images]
            return model.compute(*gray).astype(np.float32)/16
        return predict, dict(opencv=cv2.__version__, max_disparity=192,
            block_size=15 if method == 'stereobm' else 5, checkpoint=None,
            implementation='OpenCV StereoBM' if method == 'stereobm' else 'OpenCV StereoSGBM 3WAY',
            runtime='CPU', weights_retrained=False)
    import torch
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    normalize = False
    padder_type = None
    if method in ['psmnet', 'raft_stereo']:
        base = STEREO / 'baseline_psmnet_raft_stereo'
        if method == 'psmnet':
            upstream = base / 'upstream/PSMNet'
            sys.path.insert(0, str(upstream))
            from models import stackhourglass
            model = stackhourglass(192)
            ck = base/'models/psmnet_kitti2015.tar'
            state = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
            normalize = True
        else:
            upstream = base/'upstream/RAFT-Stereo'
            sys.path.insert(0, str(upstream))
            from core.raft_stereo import RAFTStereo
            from core.utils.utils import InputPadder
            padder_type = InputPadder
            model = RAFTStereo(argparse.Namespace(hidden_dims=[128]*3,
                corr_implementation='reg', shared_backbone=False, corr_levels=4,
                corr_radius=4, n_downsample=2, context_norm='batch', slow_fast_gru=False,
                n_gru_layers=3, mixed_precision=False))
            ck = base/'models/raftstereo-middlebury.pth'
            state = torch.load(ck, map_location='cpu', weights_only=False)
    elif method == 'gwcnet':
        base = STEREO/'baseline_gwcnet'
        upstream = base/'upstream/GwcNet'
        sys.path.insert(0, str(upstream))
        from models import GwcNet_GC
        model = GwcNet_GC(192)
        ck = base/'models/sceneflow/gwcnet-gc/checkpoint_000015.ckpt'
        state = torch.load(ck, map_location='cpu', weights_only=False)['model']
        normalize = True
    elif method == 'igev_stereo':
        base = STEREO/'baseline_igev_stereo'
        upstream = base/'upstream/IGEV'
        sys.path.insert(0, str(base/'python_deps'))
        sys.path.insert(0, str(upstream/'IGEV-Stereo'))
        from core.igev_stereo import IGEVStereo
        from core.utils.utils import InputPadder
        import timm
        padder_type = InputPadder
        create = timm.create_model
        def local_model(*a, **kw):
            kw['pretrained'] = False
            return create(*a, **kw)
        with patch('timm.create_model', local_model):
            model = IGEVStereo(argparse.Namespace(hidden_dims=[128]*3, corr_levels=2,
                corr_radius=4, n_downsample=2, n_gru_layers=3, max_disp=192,
                mixed_precision=False, precision_dtype='float32'))
        ck = base/'models/sceneflow.pth'
        state = torch.load(ck, map_location='cpu', weights_only=False)
    else:
        base = GRASP/'baseline_graspgen'
        upstream = base/'upstream/FoundationStereo'
        sys.path.insert(0, str(upstream))
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        from omegaconf import OmegaConf
        import timm
        padder_type = InputPadder
        ck = base/'models/foundation_stereo/23-51-11/model_best_bp2.pth'
        cfg = OmegaConf.load(ck.parent/'cfg.yaml')
        cfg.vit_size = cfg.get('vit_size', 'vitl')
        cfg.valid_iters = 32
        create, hub = timm.create_model, torch.hub.load
        def local_model(*a, **kw):
            kw['pretrained'] = False
            return create(*a, **kw)
        def local_dino(repo, model_name, *a, **kw):
            assert repo == 'facebookresearch/dinov2' and kw.get('pretrained') is False
            return hub(str(upstream/'dinov2'), model_name, *a, source='local', **kw)
        with patch('timm.create_model', local_model), patch('torch.hub.load', local_dino):
            model = FoundationStereo(cfg)
        state = torch.load(ck, map_location='cpu', weights_only=False)['model']
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    del state
    model.cuda().eval()
    def predict(images):
        tensors = [torch.as_tensor(x.copy(), device='cuda').float()[None].permute(0, 3, 1, 2) for x in images]
        if normalize:
            mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
            std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
            tensors = [(x/255-mean)/std for x in tensors]
        padder = padder_type(tensors[0].shape, divis_by=32) if padder_type else None
        if padder:
            tensors = padder.pad(*tensors)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=method=='foundationstereo'):
            if method == 'psmnet':
                disparity = model(*tensors)
            elif method == 'gwcnet':
                disparity = model(*tensors)[-1]
            elif method == 'raft_stereo':
                disparity = -model(*tensors, iters=32, test_mode=True)[1]
            else:
                disparity = model(*tensors, iters=32, test_mode=True)
        if padder:
            disparity = padder.unpad(disparity)
        torch.cuda.synchronize()
        return disparity.squeeze().float().cpu().numpy()
    import subprocess
    return predict, dict(checkpoint=str(ck), checkpoint_sha256=sha(ck),
        timm_version=getattr(sys.modules.get('timm'),'__version__',None),
        timm_source=getattr(sys.modules.get('timm'),'__file__',None),
        strict_state_dict_load=True, weights_retrained=False, upstream=str(upstream),
        upstream_commit=(subprocess.run(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True, capture_output=True).stdout.strip() or None),
        upstream_source_sha256={str(p.relative_to(upstream)):sha(p) for p in sorted(upstream.rglob('*.py')) if '.git' not in p.parts},
        runtime='CUDA', gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        iterations=32 if padder_type else None, normalization='ImageNet RGB' if normalize else 'author internal',
        precision='FP16 autocast' if method=='foundationstereo' else 'FP32', tf32=False,
        checkpoint_domain={'psmnet':'KITTI2015','raft_stereo':'Middlebury',
            'gwcnet':'SceneFlow','igev_stereo':'SceneFlow'}.get(method, 'author pretrained foundation model'))


def export_cloud(out, depth, valid, images, calibration, task, source_hashes, route='stereo'):
    cam = calibration['cameras']['left']
    K, T = np.asarray(cam['K']), np.asarray(cam['world_from_camera'])
    y, x = np.indices(depth.shape)
    z = depth[valid]
    points = np.column_stack(((x[valid]-K[0, 2])*z/K[0, 0], (y[valid]-K[1, 2])*z/K[1, 1], z))
    points = points@T[:3, :3].T+T[:3, 3]
    colors = images[0][valid]
    workspace = (np.abs(points[:, 0]) < .45) & (np.abs(points[:, 1]) < .4) & (points[:, 2] > -.02) & (points[:, 2] < .3)
    points, colors = points[workspace], colors[workspace]
    lo = np.asarray(task['target_roi_world_m']['min']); hi = np.asarray(task['target_roi_world_m']['max'])
    # Known table plane and ROI only. Never read a target mesh/pose/mask here.
    lo[2] = .003
    target = points[((points >= lo) & (points <= hi)).all(1)]
    raw_target = len(target)
    if len(target) >= 16:
        distances, _ = cKDTree(target).query(target, k=16)
        local = distances[:, 1:].mean(1)
        target = target[local <= local.mean()+2*local.std()]
    np.savez_compressed(out/'cloud.npz', scene_points_m=points.astype(np.float32),
                        target_points_m=target.astype(np.float32), scene_colors=colors)
    if min(len(points), len(target)) > 128:
        metadata = dict(schema_version=1, units='m', frame_id='dataset_world',
            input_hashes=source_hashes, purpose='grasp_inference', geometry_source='perception_prediction',
            renderer_depth_used=False, object_pose_used=False, reference_convention='opencv_optical_x_right_y_down_z_forward',
            scene_scope='workspace_roi', gripper_id='franka_panda', perception_route=route,
            gripper_geometry=dict(max_width_m=.08, closing_axis='+y', approach_axis='+z', tcp_frame='panda_hand'),
            has_normals=False, has_colors=True,
            target_selection='fixed world ROI; predicted height >3mm; kNN16 mean+2std')
        save(out/'grasp_input.npz', dict(scene_points_m=points.astype(np.float32), target_points_m=target.astype(np.float32),
            scene_colors=colors, camera_origins_m=T[None, :3, 3], T_world_reference=T,
            point_camera_ids=np.zeros(len(points), dtype=np.int32)), metadata)
    return dict(scene_points=len(points), raw_target_points=raw_target, target_points=len(target),
                grasp_input_ready=min(len(points),len(target))>128)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--episodes', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frozen-manifest', type=Path,
                   help='Required for independent test inputs; verifies frozen source and input hashes')
    a = p.parse_args()
    frozen = None
    if a.frozen_manifest:
        frozen = json.loads(a.frozen_manifest.read_text())
        assert frozen['status'] == 'frozen_frontend_before_test'
        assert frozen['source_sha256']['experiments/perception.py'] == sha(__file__)
        assert a.method in frozen['perception_methods']
    a.out.mkdir(parents=True, exist_ok=False)
    np.random.seed(0)
    t = time.perf_counter()
    predictor, model_record = make_predictor(a.method)
    model_record.update(model_load_and_provenance_seconds=time.perf_counter()-t,
                        source_sha256=sha(__file__), seed=0, input_size=[640,480])
    write(a.out/'model.json', model_record)
    for index, episode in enumerate(a.episodes):
        src, out = a.inputs/episode, a.out/episode
        out.mkdir()
        started = time.perf_counter()
        task = json.loads((src/'task.json').read_text())
        if frozen:
            assert task['split'] == frozen['split'] and episode in frozen['episodes']
            for name in ['left.png', 'right.png', 'cameras.json', 'task.json']:
                assert sha(src/name) == frozen['input_sha256'][episode+'/'+name]
        else:
            assert task['split']=='development', 'Formal test requires a frozen protocol and separate launch'
        calibration = json.loads((src/'cameras.json').read_text())
        images = [cv2.cvtColor(cv2.imread(str(src/(n+'.png'))), cv2.COLOR_BGR2RGB) for n in ['left','right']]
        assert images[0].shape == images[1].shape == (480,640,3)
        before = time.perf_counter()
        def deadline(signum, frame):
            raise TimeoutError('Perception exceeded the frozen 120 s per-image budget')
        previous = signal.signal(signal.SIGALRM, deadline) if frozen else None
        if frozen:
            signal.setitimer(signal.ITIMER_REAL, frozen['budgets']['perception_timeout_wall_seconds'])
        try:
            disparity = predictor(images)
        except Exception as error:
            write(out/'perception.json', dict(episode_id=episode, method=a.method, split=task['split'],
                status='timeout' if isinstance(error, TimeoutError) else 'model_error',
                error=repr(error), perception_pipeline_seconds=time.perf_counter()-started,
                grasp_input_ready=False, geometry_truth_used=False, object_pose_used=False))
            raise
        finally:
            if frozen:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
        inference_seconds = time.perf_counter()-before
        y,x = np.indices(disparity.shape)
        valid = np.isfinite(disparity)&(disparity>0)&(x-disparity>=0)
        depth = np.zeros_like(disparity)
        depth[valid] = calibration['cameras']['left']['K'][0][0]*calibration['stereo_baseline_m']/disparity[valid]
        valid &= (depth>.15)&(depth<1.)
        hashes = {n:sha(src/n) for n in ['left.png','right.png','cameras.json','task.json']}
        cloud = export_cloud(out, depth, valid, images, calibration, task, hashes)
        np.savez_compressed(out/'prediction.npz', disparity=disparity, depth=depth, valid=valid)
        color = cv2.applyColorMap(np.uint8(np.nan_to_num(np.clip((depth-.2)/.6,0,1))*255),cv2.COLORMAP_TURBO)
        color[~valid]=0
        cv2.imwrite(str(out/'depth.png'), color)
        elapsed = time.perf_counter()-started
        record = dict(episode_id=episode, method=a.method, split=task['split'], seed=0,
            status='timeout' if frozen and elapsed > frozen['budgets']['perception_timeout_wall_seconds'] else 'completed',
            input_sha256=hashes, inference_seconds=inference_seconds,
            perception_pipeline_seconds=elapsed,
            first_inference_after_model_load=index==0, geometry_truth_used=False,
            object_pose_used=False, native_robot_execution_performed=False,
            prediction_sha256=sha(out/'prediction.npz'), **cloud)
        write(out/'perception.json', record)
        print(json.dumps(record), flush=True)


from visual_preprocessing import export_cloud

if __name__=='__main__':
    main()
