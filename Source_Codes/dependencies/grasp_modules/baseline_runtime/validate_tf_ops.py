#!/usr/bin/env python3
"""Check PointNet++ sampling/grouping against a small known point set."""
from pathlib import Path
import json
import sys
import numpy as np
import tensorflow.compat.v1 as tf

ROOT=Path(__file__).resolve().parent/'compat/contact_graspnet'
for sub in ['sampling','grouping','3d_interpolation']:
    sys.path.insert(0,str(ROOT/'pointnet2/tf_ops'/sub))
from tf_sampling import farthest_point_sample,gather_point
from tf_grouping import query_ball_point,group_point
from tf_interpolate import three_nn,three_interpolate
tf.disable_eager_execution()
assert tf.config.list_physical_devices('GPU')
values=np.array([[[0,0,0],[1,0,0],[0,2,0],[0,0,3]]],dtype=np.float32)
with tf.device('/GPU:0'):
    points=tf.constant(values)
    idx=farthest_point_sample(2,points)
    selected=gather_point(points,idx)
    neighbors,counts=query_ball_point(.1,2,points,selected)
    grouped=group_point(points,neighbors)
with tf.device('/CPU:0'):
    distances,nearest=three_nn(points,points)
    interpolated=three_interpolate(points,nearest,tf.constant([[[1.,0.,0.]]*4]))
config=tf.ConfigProto(allow_soft_placement=False)
config.gpu_options.allow_growth=True
with tf.Session(config=config) as s:
    result=s.run([idx,selected,grouped,counts,distances,interpolated])
assert result[0].tolist()==[[0,3]], result[0]
assert np.allclose(result[1],values[:,[0,3]])
assert np.allclose(result[2][:,:,0],result[1])
assert np.all(result[3]==1)
assert np.allclose(result[4][:,:,0],0)
assert np.allclose(result[5],values)
record=dict(passed=True,tensorflow=tf.__version__,sampling_device=idx.device,
    checks=['known farthest indices','gather coordinates','radius grouping','nearest interpolation'],
    build_manifest=str(ROOT/'build_manifest.json'))
(ROOT/'ops_validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
