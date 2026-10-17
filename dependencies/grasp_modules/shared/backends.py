"""Existing author models, with perception-independent input adaptation."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
MODULE_ROOT=Path(__file__).resolve().parents[1]
HERE=MODULE_ROOT/'baseline_runtime'
SOURCES=MODULE_ROOT

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def sixdof(points, camera, seed=0):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution();tf.set_random_seed(seed)
    assert tf.config.list_physical_devices('GPU')
    upstream=HERE/'compat/6dof-graspnet'
    assert (upstream/'compatibility_manifest.json').exists()
    os.chdir(upstream)
    sys.path.insert(0,str(upstream));sys.path.insert(0,str(HERE/'compat/pointnet2/utils'))
    import grasp_estimator
    models=SOURCES/'baseline_6dof_graspnet/models'
    cfg=grasp_estimator.joint_config(str(models/'ACRONYM_VAE'),str(models/'ACRONYM_Evaluator'))
    cfg.gpu=0;cfg.ngpus=1;cfg.num_samples=50;cfg.threshold=-1.
    cfg.sample_based_improvement=0;cfg.num_refine_steps=10
    cfg.grasp_selection_mode='better_than_threshold_in_sequence'
    estimator=grasp_estimator.GraspEstimator(cfg)
    estimator.build_network()
    loaded_prefixes = [Path(tf.train.latest_checkpoint(folder)) for folder in
                       [cfg.vae_checkpoint_folder, cfg.evaluator_checkpoint_folder]]
    settings=tf.ConfigProto(allow_soft_placement=False,intra_op_parallelism_threads=8,inter_op_parallelism_threads=4)
    settings.gpu_options.allow_growth=True
    pc_camera=((points-camera[:3,3])@camera[:3,:3]).astype(np.float32)
    started=time.perf_counter()
    with tf.Session(config=settings) as session:
        estimator.load_weights(session)
        latents=np.concatenate([estimator.sample_latents() for _ in range(4)])
        poses,scores,_=estimator.predict_grasps(session,pc_camera,latents,num_refine_steps=10)
    poses=camera[None]@np.asarray(poses)
    record=dict(method='6-DOF GraspNet',seconds=time.perf_counter()-started,seed=seed,
        source='Official VAE generator and evaluator; author ACRONYM release',
        training_data_note='Author later ACRONYM checkpoint; not original 2019 training split',
        gradient_refinement_steps=10,candidates=200,batch_size=50,score_threshold=-1.,
        checkpoint_prefixes=[str(p) for p in loaded_prefixes],
        checkpoint_sha256={str(p.relative_to(models)):sha(p) for prefix in loaded_prefixes
                           for p in prefix.parent.glob(prefix.name+'.*')},
        config_sha256={str(p.relative_to(models)):sha(p) for name in ['ACRONYM_VAE','ACRONYM_Evaluator']
                       for p in (models/name).glob('*.yaml')},
        tensorflow=tf.__version__,strict_checkpoint_restore=True,
        upstream_commit='cd08b6c559e51a187ab65faea6415f51239808b8',
        input_frame='OpenCV reference optical frame, centered internally by author code',
        compatibility_manifest_sha256=sha(upstream/'compatibility_manifest.json'),
        gripper_frame='Author Panda: x closing, z approach; rotate local z -90 deg to native fingers')
    return poses,scores,np.full(len(poses),.08),record


def contact(points, camera, scene_points, seed=0):
    import tensorflow.compat.v1 as tf
    import yaml
    tf.disable_eager_execution()
    tf.set_random_seed(seed)
    assert tf.config.list_physical_devices('GPU'), 'TensorFlow cannot see CUDA GPU'
    upstream = HERE / 'compat/contact_graspnet'
    assert (upstream/'build_manifest.json').is_file(), 'Build and validate TF operators first'
    sys.path.insert(0, str(upstream/'contact_graspnet'))
    from contact_grasp_estimator import GraspEstimator
    modeldir = SOURCES/'baseline_contact_graspnet/models/scene_test_2048_bs3_hor_sigma_001'
    config = yaml.safe_load((modeldir/'config.yaml').read_text())
    config['OPTIMIZER']['batch_size'] = 1
    config['DATA']['classes'] = None
    estimator = GraspEstimator(config)
    estimator.build_network()
    checkpoint = modeldir/'model.ckpt-54054'
    settings = tf.ConfigProto(allow_soft_placement=False, intra_op_parallelism_threads=8,
                             inter_op_parallelism_threads=4)
    settings.gpu_options.allow_growth = True
    pc_camera=((scene_points-camera[:3,3])@camera[:3,:3]).astype(np.float32)
    target_camera=((points-camera[:3,3])@camera[:3,:3]).astype(np.float32)
    started = time.perf_counter()
    with tf.Session(config=settings) as session:
        tf.train.Saver().restore(session, str(checkpoint))
        poses, scores, contacts, widths = estimator.predict_grasps(session,pc_camera,
                                                 convert_cam_coords=True,forward_passes=1)
    raw_count=len(poses)
    if raw_count:
        from scipy.spatial import cKDTree
        keep=cKDTree(target_camera).query(np.asarray(contacts).reshape(-1,3))[0]<.005
        poses,scores,widths=poses[keep],scores[keep],np.asarray(widths).reshape(-1)[keep]
    poses = camera[None] @ poses
    record=dict(method='Contact-GraspNet', seconds=time.perf_counter()-started, seed=seed,
        source='Author TensorFlow model and checkpoint', checkpoint=str(checkpoint),
        checkpoint_sha256={p.name:sha(p) for p in modeldir.glob(checkpoint.name+'.*')},
        tensorflow=tf.__version__, strict_checkpoint_restore=True, forward_passes=1,
        config_sha256=sha(modeldir/'config.yaml'), upstream_commit='721706067cc7f7889c32afe7b4f7955eb90bd866',
        compatibility_manifest_sha256=sha(upstream/'build_manifest.json'),
        input_frame='OpenCV reference optical frame, converted internally by author code',
        input_scene_points=len(pc_camera),raw_grasps=raw_count,target_filter_distance_m=.005,
        cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING')=='1',
        gripper_frame='Author Panda: x closing, z approach; rotate local z -90 deg to native fingers')
    return poses, scores, widths, record



def graspgen(points, count=400, seed=0):
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    root=MODULE_ROOT/'baseline_graspgen'
    sys.path.insert(0,str(root/'upstream/GraspGen'))
    sys.path.insert(0,str(root/'ops'))
    import torch
    from grasp_gen.grasp_server import GraspGenSampler,load_grasp_cfg
    from run_inference import seed_all,validate
    seed_all(seed);torch.set_num_threads(8)
    cfg=load_grasp_cfg(str(root/'models/checkpoints/graspgen_franka_panda.yml'))
    model=GraspGenSampler(cfg)
    assert next(model.model.parameters()).is_cuda
    start=time.perf_counter()
    poses,scores=GraspGenSampler.run_inference(np.asarray(points,dtype=np.float32),model,
        grasp_threshold=-1,num_grasps=count,topk_num_grasps=count,min_grasps=count,
        max_tries=1,remove_outliers=False)
    torch.cuda.synchronize()
    poses,scores=poses.cpu().numpy(),scores.cpu().numpy()
    checks=validate(poses,scores,count)
    record=dict(method='GraspGen',source='Official diffusion generator and discriminator',
        seconds=time.perf_counter()-start,candidates=count,seed=seed,pose_checks=checks,
        gripper='franka_panda',checkpoint_config_sha256=sha(root/'models/checkpoints/graspgen_franka_panda.yml'),
        checkpoint_sha256={str(p):sha(p) for p in [Path(cfg.eval.checkpoint),Path(cfg.discriminator.checkpoint)]})
    return poses,scores,np.full(len(poses),.08),record
