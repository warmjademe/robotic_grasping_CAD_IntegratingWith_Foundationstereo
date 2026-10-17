"""Numerical checks for compiled author CUDA ops, independent torch references."""
import json
from pathlib import Path
import sys
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
for p in ['pointnet2', 'knn']:
    sys.path.insert(0, str(ROOT/'upstream/graspnet-baseline'/p))
from pointnet2_utils import (furthest_point_sample, gather_operation, grouping_operation,
                             three_nn, three_interpolate, ball_query, cylinder_query)
from knn_modules import knn
torch.manual_seed(42)
x = (torch.rand(1, 97, 3, device='cuda')+2).contiguous()
q = x[:, :7].contiguous()
checks = {}
dist, idx = three_nn(q, x)
reference, ri = torch.cdist(q, x).topk(3, largest=False)
checks['three_nn_indices'] = bool(torch.equal(idx.long(), ri))
checks['three_nn_distance'] = bool(torch.allclose(dist, reference, atol=.002))
features = torch.randn(1, 5, 97, device='cuda')
gather = gather_operation(features, idx[:, :, 0].contiguous())
checks['gather'] = bool(torch.allclose(gather, features[:, :, ri[0, :, 0]]))
grouped = grouping_operation(features, idx)
checks['grouping'] = bool(torch.allclose(grouped, features[:, :, ri[0]]))
w = torch.rand(1, 7, 3, device='cuda')
w /= w.sum(-1, keepdim=True)
interp = three_interpolate(features, idx, w)
checks['interpolation'] = bool(torch.allclose(interp, (features[:, :, ri[0]]*w[:, None]).sum(-1), atol=1e-6))
fps = furthest_point_sample(x, 11)[0]
selected, nearest = [0], torch.full((97,), float('inf'), device='cuda')
for _ in range(10):
    nearest = torch.minimum(nearest, ((x[0]-x[0, selected[-1]])**2).sum(-1))
    selected.append(int(nearest.argmax()))
checks['farthest_sampling'] = fps.tolist() == selected
ball = ball_query(.4, 12, x, q)
ball_ok = True
for j in range(7):
    candidates = torch.nonzero(((x[0]-q[0, j])**2).sum(-1) < .4**2).flatten().tolist()
    expected = candidates[:12]+[candidates[0]]*max(0, 12-len(candidates))
    ball_ok &= ball[0, j].tolist() == expected
checks['ball_query'] = bool(ball_ok)
R = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], device='cuda')
rots = R.reshape(1, 1, 9).repeat(1, 7, 1)
cyl = cylinder_query(.4, -.2, .3, 12, x, q, rots)
cyl_ok = True
for j in range(7):
    local = (x[0]-q[0, j]) @ R
    mask = (local[:, 0] > -.2) & (local[:, 0] < .3) & ((local[:, 1:]**2).sum(-1) < .4**2)
    candidates = torch.nonzero(mask).flatten().tolist()
    expected = candidates[:12]+[candidates[0]]*max(0, 12-len(candidates))
    cyl_ok &= cyl[0, j].tolist() == expected
checks['rotated_cylinder_query'] = bool(cyl_ok)
ki = knn(x.transpose(1, 2).contiguous(), q.transpose(1, 2).contiguous(), 3)
# The author KNN operator deliberately returns one-based indices.
checks['knn_one_based_indices'] = bool(torch.equal(ki.transpose(1, 2)-1, ri))
torch.cuda.synchronize()
record = dict(passed=all(checks.values()), checks=checks, torch=torch.__version__,
              cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), training=False)
(ROOT/'logs/cuda_operator_validation.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
assert record['passed']
