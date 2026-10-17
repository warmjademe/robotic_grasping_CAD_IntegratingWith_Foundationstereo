"""Cold-start one inference program in an isolated Linux network namespace."""
import argparse
import errno
import json
import os
import runpy
import socket
import sys
import time
from pathlib import Path
from project_common import ROOT, sha, write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('--script',type=Path,required=True)
    p.add_argument('--host-network-namespace',required=True)
    p.add_argument('--allow-cad',action='store_true')
    p.add_argument('script_args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    assert sys.platform=='linux' and os.geteuid()!=0
    namespace=os.readlink('/proc/self/ns/net');host=a.host_network_namespace
    assert host.startswith('net:[') and host.endswith(']')
    assert namespace!=host, 'Run this entrypoint in a dedicated unshare --net namespace'
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        try:
            probe.connect(('1.1.1.1',443))
        except OSError as error:
            assert error.errno in (errno.ENETUNREACH,errno.EHOSTUNREACH), repr(error)
            unreachable=error.errno
        else:
            raise RuntimeError('External connection succeeded in an offline test')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    excluded=[a.source/'datasets/evaluation',a.source/'datasets/scene_specs']
    if not a.allow_cad:
        excluded += [a.source/'datasets/assets',ROOT/'datasets/original',ROOT/'datasets/assets',ROOT/'datasets/canonical_assets']
    excluded=[x.resolve() for x in excluded]
    access={'blocked_reads':0,'python_open_events':0}
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            access['python_open_events']+=1
            path=Path(os.fsdecode(args[0])).resolve()
            if any(path==x or x in path.parents for x in excluded):
                access['blocked_reads']+=1
                raise PermissionError('Inference attempted to read excluded evaluation information')
    state=dict(status='running',started_unix=time.time(),network_namespace=namespace,
        host_network_namespace=host,external_probe_errno=unreachable,network_isolated=True,
        uid=os.geteuid(),pid=os.getpid(),script=str(a.script),script_sha256=sha(a.script),
        cad_allowed=a.allow_cad,file_guard_scope='Python audit events; native I/O also requires code and source audit',
        counts_as_physical_grasp=False)
    if a.record.exists():raise RuntimeError('Keep prior offline attempt; choose a new record path after diagnosis')
    write(a.record,state)
    sys.addaudithook(guard)
    args=a.script_args[1:] if a.script_args[:1]==['--'] else a.script_args
    sys.path.insert(0,str(a.source/'code'))
    sys.path.insert(0,str(a.script.resolve().parent))
    sys.argv=[str(a.script),*args]
    try:
        runpy.run_path(str(a.script),run_name='__main__')
    except BaseException as error:
        state.update(status='error',error_type=type(error).__name__,error=str(error))
        raise
    else:
        state['status']='completed'
    finally:
        state.update(finished_unix=time.time(),access=access)
        write(a.record,state)


if __name__=='__main__':main()
