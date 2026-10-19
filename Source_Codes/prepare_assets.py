"""Restore the frozen metric CAD support frames from separately acquired models.

This script uses the archived CAD-only support transforms. It does not inspect
grasp labels, infer poses from evaluation truth, or change object size.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True,
                   help='Directory containing tless/models_cad and itodd/models')
    p.add_argument('--workspace', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import trimesh
    from project_common import sha, write
    root = Path(__file__).resolve().parent
    catalog = json.loads((root/'datasets/canonical_catalog.json').read_text())
    records = []
    for item in catalog:
        relative = Path(item['source']).relative_to('datasets/original')
        source = args.original/relative
        assert sha(source) == item['source_sha256'], 'Original CAD hash differs: '+str(relative)
        mesh = trimesh.load(source, force='mesh', process=False)
        mesh.apply_scale(.001)
        points, faces = trimesh.sample.sample_surface(mesh, 20000, seed=item['bop_id'])
        normals = mesh.face_normals[faces]
        # The original pipeline exported its metric mesh before computing the
        # support frame. Reproduce that float32 PLY round-trip and the two
        # transforms separately, so canonical mesh bytes are identical.
        original = args.workspace/item['original_mesh']
        original.parent.mkdir(parents=True, exist_ok=True)
        if not original.exists():mesh.export(original)
        assert sha(original)==item['original_mesh_sha256']
        mesh=trimesh.load(original, force='mesh', process=False)
        hull=mesh.convex_hull
        transforms, probabilities=trimesh.poses.compute_stable_poses(hull,
            center_mass=hull.center_mass, sigma=0.0, n_samples=1, threshold=0.0)
        best=sorted(range(len(transforms)), key=lambda i:(-round(float(probabilities[i]),12),
            tuple(np.round(transforms[i].ravel(),10))))[0]
        transform=transforms[best].copy()
        mesh=mesh.copy()
        mesh.apply_transform(transform)
        shift=-np.r_[mesh.bounds.mean(axis=0)[:2],mesh.bounds[0,2]]
        transform[:3,3]+=shift
        mesh.apply_translation(shift)
        assert np.allclose(transform,item['support_from_original_cad'],atol=1e-12,rtol=0)
        points = points@transform[:3, :3].T+transform[:3, 3]
        normals = normals@transform[:3, :3].T
        dest = args.workspace/item['mesh']
        samples = dest.with_name('cad_samples.npz')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or samples.exists():
            assert dest.exists() and samples.exists(), 'Preserve incomplete asset reconstruction'
        else:
            mesh.export(dest)
            np.savez_compressed(samples, points_m=points, normals=normals)
        assert np.allclose(trimesh.load(dest, force='mesh', process=False).bounds,
                           item['bounds_m'], atol=1e-7, rtol=0)
        # Serialization hashes can depend on library versions. Require exact
        # frozen bytes for this reproduction instead of silently substituting.
        assert sha(dest) == item['mesh_sha256'], 'Canonical mesh differs; check trimesh version'
        assert sha(samples) == item['samples_sha256'], 'CAD samples differ; check numpy/trimesh versions'
        records.append(dict(object_id=item['object_id'], mesh_sha256=sha(dest), samples_sha256=sha(samples)))
    write(args.workspace/'assets_verified.json', dict(status='passed', objects=records))
    print('ASSETS_VERIFIED', len(records))


if __name__ == '__main__':
    main()
