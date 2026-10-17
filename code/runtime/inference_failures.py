"""Known algorithm failures terminate a task; infrastructure errors propagate."""
from contextlib import contextmanager
import signal
import time
import traceback

TERMINAL_ALGORITHM_STATUSES={'algorithm_timeout','invalid_prediction','upstream_algorithm_failure'}


class AlgorithmFailure(Exception):
    status='invalid_prediction'


class AlgorithmTimeout(AlgorithmFailure):
    status='algorithm_timeout'


class InvalidPrediction(AlgorithmFailure):
    status='invalid_prediction'


@contextmanager
def algorithm_deadline(seconds):
    """POSIX inference budget; native calls may deliver the alarm on return.

    A post-call elapsed check also discards results arriving after the budget.
    A native call that never returns remains an unresolved infrastructure job
    for the supervisor; it is never silently rerun or reported as completed.
    """
    assert seconds>0
    previous=signal.getsignal(signal.SIGALRM)
    assert signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'Do not replace another active deadline'
    def expired(*_):raise AlgorithmTimeout('Frozen inference wall-time budget exceeded')
    started=time.monotonic();signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
        if time.monotonic()-started>seconds:expired()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def failure_record(error,stage,started,budget_seconds):
    assert isinstance(error,AlgorithmFailure)
    return dict(status=error.status,failure_stage=stage,error=repr(error),traceback=traceback.format_exc(),
        seconds=time.perf_counter()-started,budget_seconds=budget_seconds,terminal_algorithm_failure=True,
        automatic_retry=False,counts_as_success=False)
