"""Execute a newly prepared batch with the released frozen numerical code.

Use --list first. This release adapter schedules the original stage functions;
it never resumes, modifies, or imports the authors' historical experiment queues.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
from project_common import read, sha, write
from prepare_experiment import METHODS, STEREO

RELEASE=Path(__file__).resolve().parents[1]


def commands(run, runtime, ablation):
    python=runtime/'.venv-infer/bin/python'; code=run/'code'
    def offline(name, script, arguments=(), cad=False):
        variables=['CAD_RELEASE_ROOT','CAD_WORKSPACE','GRASPGEN_RUNTIME','GRASP_MODULE_ROOT','STEREO_MODULE_ROOT','CAD_GPU_LOCK']
        # sudo may clear the environment. Only pass the explicit path contract.
        environment=[k+'='+os.environ[k] for k in variables]
        environment+=['HOME='+str(Path.home()),'XDG_CACHE_HOME='+str(Path.home()/'.cache')]
        return ['sudo','-n','unshare','--net','--','setpriv','--reuid='+str(os.getuid()),
                '--regid='+str(os.getgid()),'--init-groups','env',*environment,str(python),'-u',
                str(RELEASE/'code/offline_entrypoint.py'),'--source',str(run),
                '--record',str(run/'ops'/('offline_'+name+'.json')),'--script',str(script),
                '--host-network-namespace',os.readlink('/proc/self/ns/net') if Path('/proc/self/ns/net').exists() else 'net:[LINUX_REQUIRED]',
                *(['--allow-cad'] if cad else []),'--',*map(str,arguments)]
    yield 'calibration_mujoco',[python,code/'physics_batch.py','--platform','franka_mujoco','--calibration-only']
    yield 'calibration_isaac',[python,code/'physics_batch.py','--platform','franka_isaac','--calibration-only']
    yield 'calibration_audit',[python,RELEASE/'code/audit_physical_records.py','--run',run,'--calibration-only']
    if not ablation:
        yield 'render',[runtime/'.venv-sim/bin/python',code/'render_stereo_isaac.py']
    yield 'validate',[python,code/'validate_dataset.py']
    yield 'freeze_inputs',[python,code/'freeze_inputs.py']
    if ablation:
        yield 'source_clouds',offline('source_clouds',RELEASE/'code/rq3_ablation_clouds.py',['--run',run],True)
        methods=[dict(perception=p,grasp='graspgen') for p in ('registered_cad','direct_concat')]
    else:
        yield 'forward',offline('forward',code/'perceive.py',['--method','foundationstereo'])
        yield 'reverse',offline('reverse',code/'reverse_stereo.py',['--source',run,'--output',run/'results/reverse_stereo'])
        yield 'fusion',offline('fusion',code/'adaptive_fusion_v6.py',['--source',run,'--reverse',run/'results/reverse_stereo',
                    '--output',run/'results/perception/cad_fused','--config',run/'protocols/adaptive_v6.json'],True)
        yield 'export',[python,RELEASE/'code/export_fused_inputs.py','--source',run]
        for method in STEREO:
            if method!='foundationstereo':
                yield 'perception_'+method,offline('perception_'+method,RELEASE/'code/single_run_perception.py',['--source',run,'--method',method])
        methods=METHODS
    metrics=RELEASE/'code'/('rq3_ablation_metrics.py' if ablation else 'single_run_metrics.py')
    yield 'geometry',[python,metrics,'geometry','--run',run]
    for method in methods:
        name='grasp_'+method['perception']+'_'+method['grasp']
        yield name,offline(name,code/'generate_grasps.py',['--perception',method['perception'],'--grasp',method['grasp']])
    yield 'manifest',[python,code/'refresh_manifest.py']
    yield 'planning',offline('planning',code/'plan_batch.py')
    for platform in ('franka_mujoco','franka_isaac'):
        yield 'physics_'+platform,[python,code/'physics_batch.py','--platform',platform]
    yield 'audit',[python,metrics,'audit','--run',run]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--list',action='store_true',help='Print stage commands; execute nothing')
    parser.add_argument('--stage',help='Execute exactly one named stage; default executes all stages')
    parser.add_argument('--timeout',type=int,default=21600,help='Per-stage infrastructure limit in seconds; algorithm budgets remain frozen')
    args=parser.parse_args();run=args.run.resolve()
    runtime=Path(os.environ['GRASPGEN_RUNTIME']).resolve()
    os.environ['CAD_RELEASE_ROOT']=str(RELEASE)
    os.environ.setdefault('CAD_WORKSPACE',str(run.parents[2]))
    os.environ.setdefault('GRASP_MODULE_ROOT',str(RELEASE/'dependencies/grasp_modules'))
    os.environ.setdefault('STEREO_MODULE_ROOT',str(RELEASE/'dependencies/stereo_modules'))
    os.environ.setdefault('CAD_GPU_LOCK',str(Path(os.environ['CAD_WORKSPACE'])/'gpu.lock'))
    Path(os.environ['CAD_GPU_LOCK']).parent.mkdir(parents=True,exist_ok=True)
    ablation='registered_cad' in read(run/'protocols/protocol.json')['stereo_methods']
    stages=list(commands(run,runtime,ablation))
    if args.stage:
        assert args.stage in dict(stages),'Unknown stage'
        stages=[s for s in stages if s[0]==args.stage]
    if args.list:
        import shlex
        for name,cmd in stages:print(name,shlex.join(map(str,cmd)))
        return
    assert os.name=='posix' and Path('/proc/self/ns/net').exists(),'Linux required'
    prepared=read(run/'ops/prepared.json')
    assert prepared['protocol_sha256']==sha(run/'protocols/protocol.json')
    for name,digest in prepared['code_sha256'].items():assert sha(run/'code'/name)==digest,name
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    for name,cmd in stages:
        marker=run/'ops'/('release_'+name+'.json')
        assert not marker.exists(),'Existing attempt retained; inspect it before explicit recovery: '+str(marker)
        assert not (run/'ops/user_stop.json').exists(),'User stop marker'
        state=dict(stage=name,status='running',command=list(map(str,cmd)),started_unix=time.time())
        write(marker,state)
        with (run/'logs'/('release_'+name+'.log')).open('x') as log:
            child=subprocess.Popen(list(map(str,cmd)),cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=child.wait(timeout=args.timeout)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                state.update(status='interrupted',finished_unix=time.time());write(marker,state)
                raise
        state.update(status='completed' if rc==0 else 'error',returncode=rc,finished_unix=time.time());write(marker,state)
        if rc:raise RuntimeError('Stage failed; retained log: '+str(run/'logs'/('release_'+name+'.log')))
        print('STAGE_COMPLETED',name,flush=True)


if __name__=='__main__':main()
