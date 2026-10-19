#!/usr/bin/env python3
"""Prepare auditable TF1 -> TF2 compat.v1 source copies for inference only."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'baseline_6dof_graspnet/upstream/6dof-graspnet'
DEST=HERE/'compat/6dof-graspnet'
assert not DEST.exists(), DEST
shutil.copytree(SOURCE,DEST,ignore=shutil.ignore_patterns('.git','__pycache__'))
changes=[]
for p in DEST.rglob('*.py'):
    old=p.read_text(); new=old.replace('import tensorflow as tf','import tensorflow.compat.v1 as tf')
    if p.name=='train.py':
        new=new.replace('    import horovod.tensorflow as hvd','    hvd = None  # No distributed training in this inference adapter')
        new=new.replace('from tensorflow.python.training.summary_io import SummaryWriterCache',
                        'from tensorflow.python.summary.writer.writer_cache import FileWriterCache as SummaryWriterCache')
    if p.name=='grasp_estimator.py':
        new=new.replace("vars(yaml.load(open(os.path.join(vae_folder, 'args.yaml'))))",
                        "vars(yaml.unsafe_load(open(os.path.join(vae_folder, 'args.yaml'))))")
        new=new.replace("vars(yaml.load(open(os.path.join(evaluator_folder, 'args.yaml'))))",
                        "vars(yaml.unsafe_load(open(os.path.join(evaluator_folder, 'args.yaml'))))")
    if new!=old:
        p.write_text(new)
        changes.append(dict(path=str(p.relative_to(DEST)),before_sha256=hashlib.sha256(old.encode()).hexdigest(),
                            after_sha256=hashlib.sha256(new.encode()).hexdigest()))
pointnet=HERE/'compat/pointnet2'
target=HERE/'compat/contact_graspnet/pointnet2'
assert (target.parent/'ops_validation.json').exists()
if not pointnet.exists(): pointnet.symlink_to(target, target_is_directory=True)
assert pointnet.resolve()==target.resolve()
(DEST/'compatibility_manifest.json').write_text(json.dumps(dict(changes=changes,
    purpose='Inference only; original neural layers and checkpoint variables retained',
    original_commit='cd08b6c559e51a187ab65faea6415f51239808b8',
    pointnet_source='Contact-GraspNet bundled PointNet++ with TF2-compatible layers',
    configuration_loader='Only author-downloaded argparse Namespace YAML files'),indent=2)+'\n')
print('PREPARED',DEST)
