"""Execute the configured Panda tasks with shared nominal physics and audited reuse."""
import argparse
import csv
import fcntl
import json
import os
import subprocess
import time
from common import RUN, PROJECT, RUNTIME, GPU_LOCK, PROTOCOL, write, sha
from summarize import measured

PLATFORMS={'franka_isaac':('run_isaac.py','.venv-sim',True),
    'franka_mujoco':('run_mujoco.py','.venv-mujoco',False)}


def resource_wait(platform):
    if platform=='franka_mujoco':return
    while True:
        probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
        desktop={'/usr/libexec/gnome-remote-desktop-daemon','/usr/bin/gnome-shell','/usr/lib/xorg/Xorg','/usr/bin/gnome-text-editor'}
        jobs=[row for row in csv.reader(probe.stdout.splitlines()) if len(row)<3 or row[1].strip() not in desktop]
        with GPU_LOCK.open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);available=True
            except BlockingIOError:available=False
        if available and probe.returncode==0 and not jobs:return
        write(RUN/'ops/physics_wait.json',dict(platform=platform,status='waiting_for_idle_gpu',updated_unix=time.time()))
        time.sleep(15)


def results(folder,platform):
    files=[folder/'result.json'] if platform!='franka_mujoco' else sorted(folder.glob('*/result.json'))
    return [r for p in files if p.exists() for r in json.loads(p.read_text())['rows']]


def execute(platform,ids,folder):
    expected=set(PROTOCOL['calibration'] if ids==['calibration'] else ids)
    old=results(folder,platform)
    if old:
        assert {r['task_id'] for r in old}==expected, f'Incomplete existing batch: {folder}'
        return old
    if (folder/'job.json').exists():raise RuntimeError(f'Unresolved prior attempt: {folder}')
    resource_wait(platform)
    script,env,is_gpu=PLATFORMS[platform]
    command=[str(RUNTIME/env/'bin/python'),'-u',str(RUN/'code'/script),'--tasks',*ids,'--profile','nominal','--output',str(folder)]
    if is_gpu:command+=['--friction','1.0']
    folder.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',platform=platform,profile='nominal',tasks=ids,started_unix=time.time(),
        code_sha256=sha(RUN/'code'/script),protocol_sha256=sha(RUN/'protocols/protocol.json'))
    write(folder/'job.json',state)
    envvars=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (folder/'worker.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=envvars)
        state['pid']=process.pid;write(folder/'job.json',state);rc=process.wait()
    rows=results(folder,platform);good=rc==0 and {r['task_id'] for r in rows}==expected
    state.update(status='completed' if good else 'infrastructure_error',returncode=rc,finished_unix=time.time(),observed=len(rows))
    write(folder/'job.json',state)
    if not good:raise RuntimeError(f'Physical worker failure; preserved {folder}')
    return rows


def run_platform(platform,calibration_only=False):
    rows=execute(platform,['calibration'],RUN/'calibration'/platform/'nominal')
    by={r['task_id']:r for r in rows};assert set(by)==set(PROTOCOL['calibration'])
    assert all(r['execution_success']==r['expected'] for r in rows), 'Calibration failed: '+platform
    assert not by['collision_blocked']['collision_free']
    write(RUN/'calibration'/platform/'nominal/gate.json',dict(passed=True,case_count=4,protocol_sha256=sha(RUN/'protocols/protocol.json')))
    if calibration_only:return
    tasks=json.loads((RUN/'inputs/manifest.json').read_text())['tasks']
    tasks=[t for t in tasks if platform in t['platforms']]
    planroot=RUN/'results/plans'
    assert all((planroot/(t['task_id']+'.json')).exists() for t in tasks)
    selected=[t for t in tasks if json.loads((planroot/(t['task_id']+'.json')).read_text())['status']=='sampled_path_found']
    # Freeze dispatch once; restarting cannot renumber batches or repeat reuse.
    queuefile=RUN/'ops'/f'{platform}_queue.json'
    if queuefile.exists():ids=json.loads(queuefile.read_text())['task_ids']
    else:
        done=measured();ids=[t['task_id'] for t in selected if (platform,'nominal',t['task_id']) not in done]
        write(queuefile,dict(task_ids=ids,selected_total=len(selected),reused=len(selected)-len(ids)))
    for i in range(0,len(ids),8):
        execute(platform,ids[i:i+8],RUN/'results/physics'/platform/'nominal'/f'batch_{i//8:04d}')
        write(RUN/'ops'/f'{platform}_status.json',dict(status='running',executed_new=min(i+8,len(ids)),new_required=len(ids),selected_total=len(selected),denominator=len(tasks)))
        subprocess.run([str(RUNTIME/'.venv-infer/bin/python'),str(RUN/'code/summarize.py')],check=True)
    write(RUN/'ops'/f'{platform}_complete.json',dict(status='completed',selected=len(selected),new_executions=len(ids),tasks=len(tasks)))
    write(RUN/'ops'/f'{platform}_status.json',dict(status='completed',executed_new=len(ids),selected_total=len(selected),denominator=len(tasks)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--platform',choices=list(PLATFORMS));p.add_argument('--calibration-only',action='store_true');a=p.parse_args()
    if (RUN/'ops/user_stop.json').exists():raise SystemExit('Stopped by user')
    if a.platform:
        with (RUN/'ops'/f'{a.platform}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run_platform(a.platform,a.calibration_only)
        return
    with (RUN/'logs/mujoco_queue.log').open('a') as log:
        cpu=subprocess.Popen([str(RUNTIME/'.venv-infer/bin/python'),'-u',str(RUN/'code/physics_batch.py'),'--platform','franka_mujoco'],stdout=log,stderr=subprocess.STDOUT)
        error=None
        try:
            for platform in ['franka_isaac']:run_platform(platform)
        except BaseException as e:
            error=e;write(RUN/'ops/physics_error.json',dict(error=repr(e),status='error_draining_cpu',updated_unix=time.time()))
        rc=cpu.wait()
        if error:raise error
        if rc:raise RuntimeError('MuJoCo queue failed; see logs/mujoco_queue.log')

if __name__=='__main__':main()
