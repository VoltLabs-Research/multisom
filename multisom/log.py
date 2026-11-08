"""Execution-log lines for the VOLT panel, shared by the whole plugin.

STDOUT carries the persistent host's binary protocol, so every line goes to
STDERR. The helpers mirror coretoolkit's ``Volt::formatDuration``,
``Volt::Stopwatch`` and ``Volt::ProgressReporter`` so that Python and native
plugins read alike in the panel.
"""

import sys
import time

from multisom.progress import job_progress

_NEVER = float('inf')


def log(message):
    """Emit one line to STDERR (STDOUT is reserved by the host)."""
    print(message, file=sys.stderr, flush=True)


def warn(message):
    log('Warning: ' + message)


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


class Progress:
    """Percentage lines for a long serial loop that cannot flood the panel.

    At most one line per ``step_percent`` of the work and per ``min_interval``
    seconds, counted from construction, so a loop that ends within
    ``min_interval`` prints nothing; the 100% line is left to the caller's own
    summary. Call ``update(i + 1)`` once per iteration: until the next step is
    reached it costs a single compare. A ``label`` of None keeps it silent.
    """

    def __init__(self, label, total, unit='', step_percent=10, min_interval=1.0):
        self.label = label
        self.total = int(total) if label is not None else 0
        self._suffix = ' ' + unit if unit else ''
        self._step = min(max(int(step_percent), 1), 50)
        self._min_interval = min_interval
        self._start = self._last_line = time.monotonic()
        self._next_mark = self._mark_after(0)
        # The pass also fills the run's current progress step, about 200 times over.
        self._forward_step = max(1, self.total // 200)
        self._next_forward = (min(self.total, self._forward_step)
                              if job_progress.enabled and self.total > 0 else _NEVER)
        if self._next_forward != _NEVER:
            job_progress.set_text(label)

    def _mark_after(self, done):
        # Item count at which the step after the one ``done`` sits in begins.
        if self.total <= 0:
            return _NEVER
        next_percent = (done * 100 // self.total // self._step + 1) * self._step
        if next_percent >= 100:
            return _NEVER
        return -(-self.total * next_percent // 100)

    def update(self, done):
        if done >= self._next_forward:
            self._forward(done)
        if done >= self._next_mark:
            self._report(done)

    def _forward(self, done):
        self._next_forward = _NEVER if done >= self.total else min(self.total, done + self._forward_step)
        job_progress.set_fraction(min(done, self.total) / self.total)

    def _report(self, done):
        self._next_mark = self._mark_after(done)
        if done >= self.total:
            return
        now = time.monotonic()
        if now - self._last_line < self._min_interval:
            return
        self._last_line = now
        log('%s: %d%% (%d/%d%s)' % (self.label, done * 100 // self.total, done, self.total, self._suffix))

    def elapsed(self):
        return format_duration(time.monotonic() - self._start)
