"""Confidence-limited CAD correction using reliable visible-depth evidence.

The error scales and confidences are external estimates, not calibrated model
probabilities. Geometry truth and grasp outcomes are evaluation-only inputs.
"""
import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import geometry_v6 as geom
import refined_v6 as refined
from fusion_common import ROOT,SOURCE,read,write,sha


def surface_evidence(points,cameras,pred,qdepth,uncertainty,cfg):
    """Separate measured support, foreground occlusion and free-space conflict."""
    usable=np.ones(len(points),bool);views={}
    for side in ['left','right']:
        uv,z,inside=geom.project(points,cameras[side],pred[side]['depth'].shape)
        ix=np.where(inside)[0]
        measured=pred[side]['depth'][uv[ix,1],uv[ix,0]]
        sigma=uncertainty[side][uv[ix,1],uv[ix,0]]
        tolerance=np.clip(3*sigma,cfg['surface_tolerance_floor_m'],cfg['free_space_tolerance_m'])
        reliable=pred[side]['valid'][uv[ix,1],uv[ix,0]].copy()
        reliable &= qdepth[side][uv[ix,1],uv[ix,0]]>=cfg['minimum_depth_confidence_for_free_space']
        reliable &= np.isfinite(measured)&np.isfinite(sigma)
        conflict=reliable&(z[ix]<measured-tolerance)
        usable[ix[conflict]]=False
        # Use only the CAD front surface in each pixel for support statistics.
        visible=geom.visible_indices(points,cameras[side],pred[side]['depth'].shape)
        location=np.searchsorted(ix,visible)
        valid=reliable[location];delta=z[visible]-measured[location]
        occluded=valid&(delta>tolerance[location])
        support=valid&(np.abs(delta)<=tolerance[location])
        denominator=int((valid&~occluded).sum())
        views[side]=dict(visible_points=len(visible),reliable_visible_points=int(valid.sum()),
            supported_points=int(support.sum()),occluded_points=int(occluded.sum()),
            support_fraction=float(support.sum()/denominator) if denominator else 0.,
            reliable_unoccluded_points=denominator,
            reliable_visible_fraction=float(valid.mean()) if len(valid) else 0.,
            support_fraction_all_unoccluded=float(support.sum()/(~occluded).sum()) if (~occluded).any() else 0.,
            free_space_conflicts=int(conflict.sum()))
    return usable,views


def fallback_reasons(best,qc,views,conflict_fraction,registration,fusion):
    if best is None:return ['insufficient_observations']
    reasons=[]
    if qc<registration['minimum_confidence']:reasons.append('low_registration_confidence')
    if best['overlap']<registration['minimum_overlap']:reasons.append('low_overlap')
    if best['ambiguity_confidence']<registration['minimum_ambiguity_confidence']:
        reasons.append('distinct_hypothesis_ambiguity')
    weak_colour=best['visible_foreground_agreement']<registration['minimum_foreground_agreement']
    geometry_supported=(max(v['support_fraction'] for v in views.values())>=registration['minimum_overlap']
        and sum(v['supported_points'] for v in views.values())>=registration['minimum_observations'])
    if weak_colour and not geometry_supported:reasons.append('weak_image_and_surface_support')
    if conflict_fraction>fusion['maximum_free_space_conflict_fraction']:
        reasons.append('excessive_free_space_conflict')
    return reasons


def fuse_observations(clean,qv,sigma_z,registered,qc,cfg):
    distance,nearest=cKDTree(registered).query(clean)
    inlier=distance<=cfg['correspondence_sigma_m']
    weights=np.maximum(qv[inlier],.05)
    residual_scale=(float(np.sqrt(np.average(distance[inlier]**2,weights=weights)))
        if inlier.any() else cfg['correspondence_sigma_m'])
    sigma_cad=max(cfg['registration_sigma_floor_m'],residual_scale)/np.sqrt(max(qc,1e-8))
    sigma_visual=np.maximum(sigma_z,cfg['visual_sigma_floor_m'])/np.sqrt(np.maximum(qv,.05))
    alpha=sigma_visual**2/(sigma_visual**2+sigma_cad**2)
    alpha*=np.exp(-.5*(distance/cfg['correspondence_sigma_m'])**2)
    # Reliable observations constrain the correction even when CAD residuals
    # underestimate registration uncertainty. At q_v=1 the point is unchanged.
    alpha*=1-np.clip(qv,0,1)
    alpha=np.clip(alpha,0,cfg['max_cad_weight'])
    keep=inlier|(qv>=cfg['minimum_depth_confidence_for_free_space'])
    corrected=clean[keep]*(1-alpha[keep,None])+registered[nearest[keep]]*alpha[keep,None]
    return corrected,((1-alpha)*qv+alpha*qc)[keep],dict(
        registration_residual_scale_m=residual_scale,registration_sigma_estimate_m=float(sigma_cad),
        removed_low_confidence_noncorresponding_points=int((~keep).sum()),
        alpha_mean=float(alpha[keep].mean()) if keep.any() else 0.,
        alpha_min=float(alpha[keep].min()) if keep.any() else 0.,
        alpha_max=float(alpha[keep].max()) if keep.any() else 0.)


def process(source,reverse,output,config,sid):
    inp=source/'datasets/inputs'/sid;base=source/'results/perception/foundationstereo'/sid
    task=read(inp/'task.json');cameras=read(inp/'cameras.json')['cameras']
    meshpath=source/'datasets/assets'/task['object_id']/'mesh_m.ply'
    samples=meshpath.with_name('cad_samples.npz')
    files=[inp/'left.png',inp/'right.png',inp/'cameras.json',inp/'task.json',base/'cloud.npz',
        base/'prediction.npz',reverse/sid/'prediction.npz',meshpath,Path(geom.__file__),Path(refined.__file__)]
    if samples.exists():files.append(samples)
    hashes={str(p):sha(p) for p in files};out=output/sid
    if (out/'record.json').exists():
        old=read(out/'record.json');assert old['config']==config and old['input_sha256']==hashes
        assert old['code_sha256']==sha(__file__) and old['cloud_sha256']==sha(out/'cloud.npz')
        return old
    start=time.perf_counter()
    images={s:cv2.cvtColor(cv2.imread(str(inp/f'{s}.png')),cv2.COLOR_BGR2RGB) for s in ['left','right']}
    pred={}
    for side,path in [('left',base/'prediction.npz'),('right',reverse/sid/'prediction.npz')]:
        with np.load(path) as d:pred[side]={k:d[k] for k in d.files}
    qdepth,foreground,foreground_records,uncertainty,edges={},{},{},{},{}
    fb=float(cameras['left']['K'][0][0])*float(np.linalg.norm(np.array(cameras['left']['world_from_camera'])[:3,3]-np.array(cameras['right']['world_from_camera'])[:3,3]))
    for side,other,direction in [('left','right',-1),('right','left',1)]:
        foreground[side],foreground_records[side]=geom.foreground_confidence(images[side],cameras[side],
            task['target_roi_world_m'],task['known_table_height_m'],config['confidence'])
        qdepth[side],uncertainty[side],edges[side]=refined.confidence_maps(pred[side]['disparity'],
            pred[other]['disparity'],images[side],images[other],direction,foreground[side],fb,config['confidence'])
    with np.load(base/'cloud.npz') as d:observed=d['target_points_m'].astype(float);scene=d['scene_points_m']
    mesh=trimesh.load(meshpath,force='mesh',process=False)
    if samples.exists():
        with np.load(samples) as d:cad=d['points_m'];normals=d['normals']
    else:
        cad,faces=trimesh.sample.sample_surface(mesh,20000,seed=int(task['object_id'].split('_')[-1]));normals=mesh.face_normals[faces]
    support=refined.local_support(observed,config['confidence'])
    qv=geom.sample_map(observed,cameras['left'],qdepth['left'])*support
    sigma_z=geom.sample_map(observed,cameras['left'],uncertainty['left'])
    qfg=geom.sample_map(observed,cameras['left'],foreground['left'])
    eligible=(qfg>=config['confidence']['minimum_foreground_weight'])&(observed[:,2]<=np.ptp(mesh.vertices[:,2])+.01)
    clean=observed[eligible];weights=(.05+.95*qv[eligible])*qfg[eligible]
    reg,fusion=config['registration'],config['fusion'];best=None;trials=[];qc=0.;views={};fraction=0.;fusion_record={}
    if len(clean)>=reg['minimum_observations']:
        T,qc,best,trials=refined.fit_cad(cad,clean,weights,mesh.bounds,cameras,foreground,reg,normals)
        registered=geom.transform(cad,T);registered=registered[registered[:,2]>fusion['cad_surface_minimum_height_m']]
        usable,views=surface_evidence(registered,cameras,pred,qdepth,uncertainty,fusion)
        fraction=float(1-usable.mean())
        if not config.get('enable_visibility',True):usable[:]=True;fraction=0.
    reasons=fallback_reasons(best,qc,views,fraction,reg,fusion)
    fallback=best is None or (config.get('enable_fallback',True) and bool(reasons))
    if fallback:points=observed;origins=np.zeros(len(points),np.uint8);confidence=qv
    else:
        corrected,q,fusion_record=fuse_observations(clean,qv[eligible],sigma_z[eligible],registered,qc,fusion)
        if qc<fusion['minimum_completion_confidence']:usable[:]=False
        points=np.concatenate([corrected,registered[usable]])
        origins=np.r_[np.ones(len(corrected),np.uint8),np.full(usable.sum(),2,np.uint8)]
        confidence=np.r_[q,np.full(usable.sum(),qc)]
        ix=geom.voxel_indices(points,fusion['voxel_m']);points,origins,confidence=points[ix],origins[ix],confidence[ix]
    assert points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all()
    out.mkdir(parents=True,exist_ok=True);points=points.astype(np.float32)
    np.savez_compressed(out/'cloud.npz',target_points_m=points,
        scene_points_m=scene.copy() if fallback else np.concatenate([scene,points]),
        point_origin=origins,point_confidence=confidence.astype(np.float32))
    np.savez_compressed(out/'confidence.npz',**{s+'_depth_confidence':qdepth[s] for s in qdepth},
        **{s+'_depth_sigma_m':uncertainty[s] for s in uncertainty},
        **{s+'_foreground_confidence':foreground[s] for s in foreground},observed_local_support=support)
    result=dict(scene_id=sid,config=config,input_sha256=hashes,code_sha256=sha(__file__),
        seconds=time.perf_counter()-start,observed_points=len(observed),filtered_points=len(clean),fused_points=len(points),
        original_scene_points_retained=len(scene),fallback=bool(fallback),fallback_reasons=reasons,
        empty_output=len(points)==0,
        registration_confidence=qc,best_registration=best,trials=trials,
        surface_support=views,free_space_conflict_fraction=fraction,uncertainty_fusion=fusion_record,
        foreground=foreground_records,ground_truth_input=False,cloud_sha256=sha(out/'cloud.npz'))
    write(out/'record.json',result);print('FUSED_V5',sid,len(points),'fallback',fallback,flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--reverse',type=Path,default=ROOT/'results/development/reverse_stereo')
    p.add_argument('--config',type=Path,default=ROOT/'protocols/adaptive_v6.json')
    p.add_argument('--output',type=Path,default=ROOT/'results/development/adaptive_v5/perception')
    p.add_argument('--scene',action='append');a=p.parse_args()
    forbidden=[(a.source/'datasets'/n).resolve() for n in ['evaluation','scene_specs']]
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            file=Path(args[0]).resolve()
            if any(file==x or file.is_relative_to(x) for x in forbidden):raise PermissionError('Evaluation truth is not a fusion input')
    sys.addaudithook(guard)
    config=read(a.config);scenes=a.scene or [s['scene_id'] for s in read(a.source/'datasets/manifest.json')['scenes']]
    for sid in scenes:process(a.source,a.reverse,a.output,config,sid)
    write(a.output.parent/'fusion_status.json',dict(status='completed',scenes=len(scenes),config_sha256=sha(a.config),code_sha256=sha(__file__)))


if __name__=='__main__':main()
