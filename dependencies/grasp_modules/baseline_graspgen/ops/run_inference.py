"""GraspGen seed and output validation helpers used by the final experiment."""
import random
import os
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validate(poses, scores, expected):
    assert poses.shape == (expected, 4, 4), poses.shape
    assert scores.shape == (expected,), scores.shape
    assert np.isfinite(poses).all() and np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    r = poses[:, :3, :3]
    orthogonality = np.max(np.abs(r.transpose(0, 2, 1) @ r - np.eye(3)))
    determinant = np.linalg.det(r)
    assert orthogonality < 1e-4, orthogonality
    assert np.allclose(determinant, 1, atol=1e-4), determinant
    assert np.allclose(poses[:, 3, :], [0, 0, 0, 1])
    assert np.all(np.diff(scores) <= 1e-6)
    return dict(max_orthogonality_error=float(orthogonality),
                determinant_min=float(determinant.min()),
                determinant_max=float(determinant.max()))

