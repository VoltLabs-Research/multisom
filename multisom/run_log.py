"""The execution log of a plugin run, as VOLT shows it in the job's panel.

Mirrors coretoolkit's ``Volt::RunLog``. A plugin's log derives from ``RunLog``
and registers one ``dump`` per result its algorithm builds; the algorithm hands
each result to ``dump`` once the phase that built it is over, so it keeps no
timers and no strings of its own. STDOUT carries the persistent host's binary
protocol, so every line goes to STDERR. Durations read like coretoolkit's
``Volt::formatDuration``.
"""

import sys
import time
from functools import singledispatchmethod


def info(message):
    """One line of the log, on STDERR (STDOUT is reserved by the host)."""
    print(message, file=sys.stderr, flush=True)


def warn(message):
    info('Warning: ' + message)


def format_duration(seconds):
    """Elapsed wall time as a log line shows it: '420 ms', '3.18 s', '2 min 05 s'."""
    if seconds < 1.0:
        return '%.0f ms' % (seconds * 1000.0)
    if seconds < 60.0:
        return '%.2f s' % seconds
    whole = int(seconds)
    return '%d min %02d s' % (whole // 60, whole % 60)


class Stopwatch:
    def __init__(self):
        self._start = time.monotonic()

    def restart(self):
        self._start = time.monotonic()

    def seconds(self):
        return time.monotonic() - self._start

    def elapsed(self):
        return format_duration(self.seconds())


class RunLog:
    """The base of a plugin's log, which opens the run with its parameters.

    The plugin's log declares its own ``dump`` as a ``singledispatchmethod``
    whose default hands the result on to ``super().dump(result)``, so the
    results every plugin shares keep the lines registered here.
    """

    def __init__(self, name, parameters):
        self._name = name
        self._run = Stopwatch()
        self._phase = Stopwatch()
        info('Parameters: ' + ', '.join(f'{key}={value}' for key, value in parameters))

    @singledispatchmethod
    def dump(self, result):
        raise TypeError(f'{type(self).__name__} has no line for a {type(result).__name__}')

    @dump.register
    def _(self, result: dict):
        # The run's result is the last thing it builds, so its lines close the run with the total.
        atoms_path, summary_path = result['outputs']
        info(f'Exported atoms data to: {atoms_path}')
        info(f'Summary parquet written to {summary_path}')
        info(f'{self._name} completed in {self.run_elapsed()}.')

    def phase_elapsed(self):
        """How long the phase that just ended took: the time since the previous call, which starts the next phase."""
        elapsed = self._phase.elapsed()
        self._phase.restart()
        return elapsed

    def run_elapsed(self):
        return self._run.elapsed()
