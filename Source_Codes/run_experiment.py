"""Execute a prepared reproduction batch; --list prints commands without execution."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
from project_common import read, sha, write

RELEASE = Path(__file__).resolve().parent


def commands(run, runtime):
    protocol = read(run / 'protocols/protocol.json')
    profile = protocol['release_profile']
    tasks = read(run / 'inputs/manifest.json')['tasks']
    python = runtime / '.venv-infer/bin/python'
    code = run / 'code'

    def offline(name, script, arguments=(), cad=False):
        variables = ['CAD_RELEASE_ROOT', 'CAD_WORKSPACE', 'GRASPGEN_RUNTIME',
                     'GRASP_MODULE_ROOT', 'STEREO_MODULE_ROOT', 'CAD_GPU_LOCK', 'PYTHONPATH']
        environment = [k + '=' + os.environ[k] for k in variables]
        environment += ['HOME=' + str(Path.home()), 'XDG_CACHE_HOME=' + str(Path.home() / '.cache')]
        return ['sudo', '-n', 'unshare', '--net', '--', 'setpriv',
                '--reuid=' + str(os.getuid()), '--regid=' + str(os.getgid()), '--init-groups',
                'env', *environment, str(python), '-u', str(RELEASE / 'offline_entrypoint.py'),
                '--source', str(run), '--record', str(run / 'ops' / ('offline_' + name + '.json')),
                '--script', str(script), '--host-network-namespace',
                os.readlink('/proc/self/ns/net') if Path('/proc/self/ns/net').exists() else 'net:[LINUX_REQUIRED]',
                *(['--allow-cad'] if cad else []), '--', *map(str, arguments)]

    if profile != 'rq1':
        for platform in protocol['platforms']:
            yield 'calibration_' + platform, [python, code / 'physics_batch.py',
                '--platform', platform, '--calibration-only']
        yield 'calibration_audit', [python, RELEASE / 'audit_physical_records.py', '--run', run, '--calibration-only']
    yield 'render', [runtime / '.venv-sim/bin/python', code / 'render_stereo_isaac.py']
    yield 'validate', [python, code / 'validate_dataset.py']
    yield 'freeze_inputs', [python, code / 'freeze_inputs.py']
    yield 'forward', offline('forward', code / 'perceive.py', ['--method', 'foundationstereo'])
    yield 'reverse', offline('reverse', code / 'reverse_stereo.py',
        ['--source', run, '--output', run / 'results/reverse_stereo'])
    yield 'fusion', offline('fusion', code / 'adaptive_fusion_v6.py',
        ['--source', run, '--reverse', run / 'results/reverse_stereo',
         '--output', run / 'results/perception/cad_fused', '--config', run / 'protocols/adaptive_v6.json'], True)
    yield 'export', [python, RELEASE / 'export_fused_inputs.py', '--source', run]
    if profile == 'rq2_cross_grasp':
        yield 'cad_input_contract', [python, RELEASE / 'RQ2/rq2_cross_grasp_inputs.py', '--run', run]
    for perception in protocol['stereo_methods']:
        if perception not in ('foundationstereo', 'cad_fused', 'registered_cad', 'direct_concat'):
            yield 'perception_' + perception, offline('perception_' + perception,
                RELEASE / 'RQ1/single_run_perception.py', ['--source', run, '--method', perception])
    if profile == 'rq3':
        yield 'source_clouds', offline('source_clouds', RELEASE / 'RQ3/rq3_ablation_clouds.py', ['--run', run], True)
    yield 'geometry', [python, RELEASE / 'evidence_metrics.py', 'geometry', '--run', run]
    if profile == 'rq1':
        return
    combinations = sorted({(t['perception'], t['grasp']) for t in tasks})
    for perception, grasp in combinations:
        name = 'grasp_' + perception + '_' + grasp
        yield name, offline(name, code / 'generate_grasps.py', ['--perception', perception, '--grasp', grasp])
    yield 'manifest', [python, code / 'refresh_manifest.py']
    yield 'planning', offline('planning', code / 'plan_batch.py')
    for platform in protocol['platforms']:
        yield 'physics_' + platform, [python, code / 'physics_batch.py', '--platform', platform]
    yield 'audit', [python, RELEASE / 'evidence_metrics.py', 'audit', '--run', run]


def environment(run, runtime):
    os.environ['CAD_RELEASE_ROOT'] = str(RELEASE)
    os.environ['GRASPGEN_RUNTIME'] = str(runtime)
    os.environ.setdefault('CAD_WORKSPACE', str(run.parents[2]))
    os.environ.setdefault('GRASP_MODULE_ROOT', str(RELEASE / 'dependencies/grasp_modules'))
    os.environ.setdefault('STEREO_MODULE_ROOT', str(RELEASE / 'dependencies/stereo_modules'))
    os.environ.setdefault('CAD_GPU_LOCK', str(Path(os.environ['CAD_WORKSPACE']) / 'gpu.lock'))
    os.environ['PYTHONPATH'] = str(RELEASE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--stage', help='Execute one stage; default executes all stages')
    parser.add_argument('--timeout', type=int, default=21600)
    args = parser.parse_args()
    run = args.run.resolve()
    runtime = Path(os.environ['GRASPGEN_RUNTIME']).resolve()
    environment(run, runtime)
    stages = list(commands(run, runtime))
    if args.stage:
        assert args.stage in dict(stages), 'Unknown stage'
        stages = [s for s in stages if s[0] == args.stage]
    if args.list:
        import shlex
        for name, command in stages:
            print(name, shlex.join(map(str, command)))
        return
    assert os.name == 'posix' and Path('/proc/self/ns/net').exists(), 'Linux required'
    prepared = read(run / 'ops/prepared.json')
    assert prepared['protocol_sha256'] == sha(run / 'protocols/protocol.json')
    assert prepared['selection_sha256'] == sha(run / 'protocols/paper_selection.json')
    assert prepared['task_identity_sha256'] == sha(run / 'protocols/task_identity.json')
    assert prepared['execution_seeds_sha256'] == sha(run / 'protocols/execution_seeds.json')
    for name, digest in prepared['code_sha256'].items():
        assert sha(run / 'code' / name) == digest, name
    Path(os.environ['CAD_GPU_LOCK']).parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    # GPU ownership belongs to each inference/render/Isaac worker. The runner
    # never holds that lock while waiting for a worker that acquires it.
    for name, command in stages:
        marker = run / 'ops' / ('release_' + name + '.json')
        assert not marker.exists(), 'Prior attempt retained; inspect before recovery: ' + str(marker)
        assert not (run / 'ops/user_stop.json').exists(), 'User stop marker'
        state = dict(stage=name, status='running', command=list(map(str, command)), started_unix=time.time())
        write(marker, state)
        with (run / 'logs' / ('release_' + name + '.log')).open('x') as log:
            child = subprocess.Popen(list(map(str, command)), cwd=run, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = child.wait(timeout=args.timeout)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                state.update(status='interrupted', finished_unix=time.time())
                write(marker, state)
                raise
        state.update(status='completed' if rc == 0 else 'error', returncode=rc, finished_unix=time.time())
        write(marker, state)
        if rc:
            raise RuntimeError('Stage failed; retained log: ' + str(run / 'logs' / ('release_' + name + '.log')))
        print('STAGE_COMPLETED', name, flush=True)


if __name__ == '__main__':
    main()
