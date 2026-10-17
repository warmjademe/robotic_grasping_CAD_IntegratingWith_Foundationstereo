"""Geometric uncertainty and normal-based CAD refinement for adaptive v2."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import geometry_v6 as initial


def confidence_maps(disparity, other, image, other_image, direction, foreground, focal_baseline, cfg):
    base,residual,photo=initial.stereo_confidence(disparity,other,image,other_image,direction,cfg)
    # These are external error estimates, not native calibrated model outputs.
    sigma_d=np.maximum(cfg['minimum_disparity_sigma_px'],residual/np.sqrt(2))
    valid=np.isfinite(disparity)&(disparity>0)
    sigma_z=np.zeros(disparity.shape,np.float32)
    sigma_z[valid]=focal_baseline*sigma_d[valid]/disparity[valid]**2
    interior=cv2.distanceTransform((foreground>=cfg['minimum_foreground_weight']).astype(np.uint8),cv2.DIST_L2,3)
    edge=cfg['edge_confidence_floor']+(1-cfg['edge_confidence_floor'])*np.clip(interior/cfg['edge_distance_px'],0,1)
    uncertainty=1/(1+(sigma_z/cfg['depth_uncertainty_scale_m'])**2)
    q=base*edge*uncertainty;q[~valid]=0
    return q.astype(np.float32),sigma_z,interior


def local_support(points,cfg):
    if len(points)<2:return np.zeros(len(points))
    k=min(cfg['local_neighbours'],len(points))
    distances=cKDTree(points).query(points,k=k)[0][:,-1]
    excess=np.maximum(distances-cfg['local_supported_radius_m'],0)
    return np.exp(-.5*(excess/cfg['local_decay_m'])**2)


def point_to_plane(cad,normals,observed,weights,T,cfg):
    tree=cKDTree(cad);T=T.copy()
    for iteration in range(cfg['point_to_plane_iterations']):
        R,t=T[:3,:3],T[:3,3]
        distance,ix=tree.query((observed-t)@R)
        use=distance<cfg['inlier_distance_m']
        if use.sum()<30:break
        rotated=cad[ix[use]]@R.T;normal=normals[ix[use]]@R.T
        residual=np.einsum('ij,ij->i',normal,observed[use]-rotated-t)
        w=weights[use]/(1+(residual/cfg['point_to_plane_robust_m'])**2)
        yaw_derivative=np.column_stack([-rotated[:,1],rotated[:,0],np.zeros(len(rotated))])
        A=np.column_stack([normal[:,:2],np.einsum('ij,ij->i',normal,yaw_derivative)])
        delta=np.linalg.lstsq(A*np.sqrt(w[:,None]),residual*np.sqrt(w),rcond=1e-6)[0]
        delta[:2]=np.clip(delta[:2],-.003,.003);delta[2]=np.clip(delta[2],-.1,.1)
        T[:2,3]+=delta[:2];T[:3,:3]=Rotation.from_euler('z',delta[2]).as_matrix()@R
        if np.linalg.norm(delta[:2])+.03*abs(delta[2])<1e-7:break
    return T,iteration+1


def score_pose(cad,observed,weights,T,cameras,foreground,cfg):
    fitted=initial.transform(cad,T)
    d=cKDTree(cad).query((observed-T[:3,3])@T[:3,:3])[0]
    forward=float(np.average(np.minimum(d,.008),weights=weights))
    tree=cKDTree(observed);reverse=[];silhouettes=[]
    for side in ['left','right']:
        vi=initial.visible_indices(fitted,cameras[side],foreground[side].shape)
        vi=vi[fitted[vi,2]>cfg['cad_surface_minimum_height_m']]
        reverse.append(float(np.minimum(tree.query(fitted[vi])[0],.008).mean()) if len(vi) else .008)
        silhouettes.append(float(initial.sample_map(fitted[vi],cameras[side],foreground[side]).mean()) if len(vi) else 0.)
    score=forward+.5*np.mean(reverse)+.003*(1-np.mean(silhouettes))
    return {'score_m':float(score),'overlap':float(np.average(d<cfg['inlier_distance_m'],weights=weights)),
        'visible_foreground_agreement':float(np.mean(silhouettes)),
        'estimated_world_from_cad':T.tolist()}


def fit_cad(cad,observed,weights,bounds,cameras,foreground,cfg,normals):
    _,_,_,trials=initial.fit_cad(cad,observed,weights,bounds,cameras,foreground,cfg)
    ix=initial.voxel_indices(observed,cfg['voxel_m'])
    if len(ix)>cfg['maximum_observed_points']:ix=ix[np.linspace(0,len(ix)-1,cfg['maximum_observed_points']).astype(int)]
    obs,w=observed[ix],weights[ix]
    refined=[]
    for old in sorted(trials,key=lambda x:x['score_m'])[:cfg['refine_hypotheses']]:
        start=np.asarray(old['estimated_world_from_cad'])
        new,iterations=point_to_plane(cad,normals,obs,w,start,cfg)
        for label,T in [('initial',start),('point_to_plane',new)]:
            refined.append(dict(score_pose(cad,obs,w,T,cameras,foreground,cfg),
                refinement=label,refinement_iterations=iterations))
    # Include all original hypotheses when checking non-equivalent alternatives.
    ranked=sorted(trials+refined,key=lambda x:x['score_m'])
    best=dict(ranked[0]);T=np.asarray(best['estimated_world_from_cad'])
    best.update(score_pose(cad,obs,w,T,cameras,foreground,cfg))
    shape=initial.transform(cad,T);tree=cKDTree(shape)
    sample=cad[np.linspace(0,len(cad)-1,min(len(cad),2048)).astype(int)]
    alternative=None;equivalent=0
    for hypothesis in ranked[1:]:
        other=np.asarray(hypothesis['estimated_world_from_cad'])
        separation=float(tree.query(initial.transform(sample,other))[0].mean())
        if separation<=cfg['equivalent_surface_distance_m']:
            equivalent+=1;continue
        alternative={'score_gap_m':max(0.,hypothesis['score_m']-best['score_m']),
                     'surface_separation_m':separation}
        break
    ambiguity=1. if alternative is None else .5+.5*np.clip(alternative['score_gap_m']/cfg['unambiguous_score_gap_m'],0,1)
    q=float(best['overlap']*np.exp(-best['score_m']/.008)*ambiguity)
    best.update(alternative=alternative,equivalent_hypotheses=equivalent,
        ambiguity_confidence=float(ambiguity),
        equivalence='CAD surface distance; geometrically equivalent symmetric orientations are grouped')
    return T,q,best,trials+refined
