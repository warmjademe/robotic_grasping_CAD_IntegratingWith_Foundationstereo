"""Explicit PTv3 input adapter; checkpoint weights and sampling stay unchanged.

Submanifold sparse convolution requires unique integer coordinates. Retain the
first observation of each occupied encoder voxel in original input order. The
encoder's configured scale/grid is used, with no additional tunable resolution.
Install only in an isolated process. This is a versioned inference change and
requires paired remeasurement of both arms before any fusion-effect claim.
"""
import numpy as np

ADAPTER_VERSION='ptv3_unique_first_v1'
RECORDS=[]


def convert(point_cloud,grid_size=.01):
    import torch
    assert point_cloud.ndim==3 and point_cloud.shape[-1]==3
    assert grid_size>0 and torch.isfinite(point_cloud).all()
    coords=[];grids=[];counts=[];details=[]
    for original in point_cloud:
        grid=torch.div(original-original.min(dim=0).values,grid_size,rounding_mode='trunc').int()
        grid_numpy=grid.detach().cpu().numpy()
        _,first=np.unique(grid_numpy,axis=0,return_index=True)
        keep=np.sort(first)
        selected=torch.as_tensor(keep,device=point_cloud.device,dtype=torch.long)
        coords.append(original[selected]);grids.append(grid[selected]);counts.append(len(keep))
        details.append({'input_points':len(original),'output_points':len(keep),
            'duplicate_indices_removed':len(original)-len(keep),'encoder_grid_size':float(grid_size)})
    RECORDS.append(details)
    coord=torch.cat(coords,dim=0)
    return dict(coord=coord,feat=coord,grid_coord=torch.cat(grids,dim=0),
        grid_size=grid_size,offset=torch.tensor(counts,device=point_cloud.device).cumsum(0))


def install():
    from grasp_gen.models import generator,discriminator
    generator.convert_to_ptv3_pc_format=convert
    discriminator.convert_to_ptv3_pc_format=convert
    return ADAPTER_VERSION
