"""Inspect/fetch pinned external source trees and verify separately acquired weights.

No files are downloaded without the explicit --fetch-sources option. Model
weights and dataset assets remain subject to their original licenses.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fetch-sources',action='store_true')
    p.add_argument('--check-weights',action='store_true')
    a=p.parse_args()
    roots={'grasp_modules':Path(os.environ.get('GRASP_MODULE_ROOT',ROOT/'dependencies/grasp_modules')),
           'stereo_modules':Path(os.environ.get('STEREO_MODULE_ROOT',ROOT/'dependencies/stereo_modules')),
           'assets':ROOT/'dependencies/assets'}
    roots['runtime']=Path(os.environ.get('GRASPGEN_RUNTIME',roots['grasp_modules']/'baseline_graspgen'))
    sources=json.loads((ROOT/'configs/upstream_sources.json').read_text())['repositories']
    for row in sources:
        dest=roots[row['group']]/row['path']
        if a.fetch_sources and not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','clone','--no-checkout',row['url'],str(dest)],check=True)
            subprocess.run(['git','-C',str(dest),'checkout','--detach',row['commit']],check=True)
            subprocess.run(['git','-C',str(dest),'submodule','update','--init','--recursive'],check=True)
        if dest.exists():
            actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
            assert actual==row['commit'],'Repository revision differs: '+row['path']
            status='revision_verified'
        else:status='not_installed'
        print(row['path'],status,row['commit'])
    if a.check_weights:
        weights=json.loads((ROOT/'configs/model_weights.json').read_text())['weights']
        for method,rows in weights.items():
            for row in rows:
                path=roots[row['group']]/row['path']
                assert path.is_file() and path.stat().st_size==row['bytes'] and digest(path)==row['sha256'], 'Weight missing or changed: '+method+'/'+row['path']
        print('WEIGHTS_VERIFIED',sum(map(len,weights.values())))


if __name__=='__main__':main()
