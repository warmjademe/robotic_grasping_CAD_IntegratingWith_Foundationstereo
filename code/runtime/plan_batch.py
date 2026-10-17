"""Plan shared Panda paths for both simulators; preserve every task status."""
import concurrent.futures
import json
import os
import time
import traceback
from common import RUN,PROTOCOL,write,sha


def work(pair):
    task,platform=pair
    assert platform=='franka'
    dest=RUN/'results/plans'/(task['task_id']+'.json')
    if dest.exists():return json.loads(dest.read_text())
    upstream=json.loads((RUN/'results/grasps'/task['task_id']/'task_result.json').read_text())
    if upstream['status']=='insufficient_points' or upstream['returned']==0:
        result=dict(task_id=task['task_id'],status=upstream['status'],selected=None,seconds=0.,platform=platform)
        write(dest,result);return result
    assert sha(RUN/task['cloud'])==task['cloud_sha256'] and sha(RUN/task['grasps'])==task['grasps_sha256']
    try:
        from plan import plan
        result=plan(task)
        result.update(task_id=task['task_id'],platform=platform)
        write(dest,result);return result
    except BaseException as error:
        write(dest.with_suffix('.error.json'),dict(task_id=task['task_id'],error=repr(error),traceback=traceback.format_exc()))
        raise


def main():
    tasks=json.loads((RUN/'inputs/manifest.json').read_text())['tasks'];counts={};start=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for index,r in enumerate(pool.map(work,[(t,p) for t in tasks for p in ['franka']])):
            key=r['platform']+':'+r['status'];counts[key]=counts.get(key,0)+1
            write(RUN/'ops/planning_status.json',dict(completed=index+1,expected=len(tasks),counts=counts,seconds=time.time()-start))
            print('PLAN',r['task_id'],r['platform'],r['status'],flush=True)
    write(RUN/'ops/planning_complete.json',dict(status='completed',tasks=len(tasks),counts=counts))

if __name__=='__main__':main()
