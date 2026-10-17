#!/usr/bin/env python3
"""Build author PointNet++ operations against the installed TensorFlow ABI.

Only compatibility copies are changed. Original sources and checkpoints remain
untouched. Build commands and source changes are recorded alongside binaries.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import tensorflow as tf

HERE = Path(__file__).resolve().parent
AUTHOR = HERE.parent / 'baseline_contact_graspnet/upstream/contact_graspnet'
DEST = HERE / 'compat/contact_graspnet'
CUDA = Path(os.environ['CUDA_HOME']).resolve()


def main():
    assert tf.__version__ == '2.15.1'
    if DEST.exists():
        if (DEST/'build_manifest.json').exists():
            raise FileExistsError('Completed build already exists: ' + str(DEST))
        DEST.rename(DEST.with_name(DEST.name+'.failed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    shutil.copytree(AUTHOR, DEST, ignore=shutil.ignore_patterns('.git', '*.so', '*.cu.o', '__pycache__'))
    patches = []
    for p in DEST.rglob('*.cpp'):
        source = p.read_text()
        changed = source.replace('Status::OK()', 'OkStatus()')
        if changed != source:
            p.write_text(changed)
            patches.append(dict(file=str(p.relative_to(DEST)), change='Status::OK() -> OkStatus()',
                upstream_sha256=hashlib.sha256(source.encode()).hexdigest()))
    commands = []
    with (DEST / 'build.log').open('w') as log:
        def run(args, cwd):
            commands.append(args)
            log.write(repr(args) + '\n'); log.flush()
            subprocess.run(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
        for folder, stem, gpu in [('sampling','tf_sampling',True),('grouping','tf_grouping',True),
                                   ('3d_interpolation','tf_interpolate',False)]:
            cwd = DEST / 'pointnet2/tf_ops' / folder
            extra = []
            if gpu:
                obj = stem + '_g.cu.o'
                run([str(CUDA/'bin/nvcc'),'-ccbin',str(CUDA/'bin/x86_64-conda-linux-gnu-g++'),'-std=c++17','-c',stem+'_g.cu','-o',obj,'-O2',
                     '-DGOOGLE_CUDA=1','-x','cu','-Xcompiler','-fPIC',
                     '-gencode=arch=compute_89,code=sm_89'], cwd)
                cuda_lib = CUDA/'targets/x86_64-linux/lib'
                extra = [obj, '-L'+str(cuda_lib), '-lcudart', '-Wl,-rpath,'+str(cuda_lib)]
            run([str(CUDA/'bin/x86_64-conda-linux-gnu-g++'),'-std=c++17','-shared',stem+'.cpp','-o',stem+'_so.so','-fPIC','-O2',
                 '-I'+str(CUDA/'targets/x86_64-linux/include'),*tf.sysconfig.get_compile_flags(),*extra,
                 *tf.sysconfig.get_link_flags(), '-Wl,-rpath,'+tf.sysconfig.get_lib()],cwd)
    record=dict(tensorflow=tf.__version__,compile_flags=tf.sysconfig.get_compile_flags(),
        link_flags=tf.sysconfig.get_link_flags(),patches=patches,commands=commands)
    (DEST/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print('BUILT', DEST, flush=True)


if __name__ == '__main__': main()
