"""Apply known gripper/workcell constraints before allocating costly IK checks."""
import numpy as np


def ordered_candidates(poses,scores,widths,valid,cfg):
    rejected=[];eligible=[]
    for rank in np.argsort(-scores,kind='stable'):
        rank=int(rank);H=poses[rank];opening=float(widths[rank]);reason=None
        if not valid[rank] or not np.isfinite(opening) or not cfg['opening_range_m'][0]<=opening<=cfg['opening_range_m'][1]:
            reason='width_invalid'
        elif H[2,2]>-np.cos(np.deg2rad(cfg['approach_cone_degrees'])):
            reason='approach_direction'
        if reason:rejected.append(dict(rank=rank,reason=reason,stage='cheap_prefilter'))
        else:eligible.append(rank)
    return np.array(eligible[:cfg['candidate_limit']],dtype=int),rejected,len(eligible)
