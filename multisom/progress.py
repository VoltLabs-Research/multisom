"""How far the run is, as VOLT shows it on the job.

Mirrors coretoolkit's ``Volt::JobProgress``: ``@volt:progress <0..1> <text>``
lines on STDERR, composed from weighted steps that may split again (the way
OVITO composes its sub-steps), so the value climbs once across every pass and
never goes back. Nothing is written unless VOLT sets ``VOLT_PROGRESS=1``.

A persistent host runs many frames in one process, so ``start()`` begins each run
from zero.
"""

import os
import sys
import time

_MIN_STEP = 0.005
_MIN_INTERVAL = 0.2


class _JobProgress:
    def __init__(self):
        self.enabled = os.environ.get('VOLT_PROGRESS') == '1'
        self.start()

    def start(self):
        self._levels = []
        self._text = ''
        self._text_changed = False
        self._reported = 0.0
        self._written = 0.0
        self._written_at = None

    def _range(self):
        low, high = 0.0, 1.0
        for base, span, bounds, index in self._levels:
            low = base + span * bounds[index]
            high = base + span * bounds[index + 1]
        return low, high

    def begin_steps(self, weights):
        if not self.enabled:
            return
        low, high = self._range()
        clean = [max(0.0, float(weight)) for weight in weights] or [1.0]
        total = sum(clean)
        bounds, cumulative = [0.0], 0.0
        for weight in clean:
            cumulative += weight
            bounds.append(cumulative / total if total > 0 else 1.0)
        self._levels.append([low, high - low, bounds, 0])

    def next_step(self):
        if not self.enabled or not self._levels:
            return
        level = self._levels[-1]
        if level[3] + 2 < len(level[2]):
            level[3] += 1
        self._publish(self._range()[0])

    def end_steps(self):
        if not self.enabled or not self._levels:
            return
        base, span, _, _ = self._levels.pop()
        self._publish(base + span)

    def set_fraction(self, fraction):
        if not self.enabled:
            return
        low, high = self._range()
        self._publish(low + min(1.0, max(0.0, float(fraction))) * (high - low))

    def set_text(self, text):
        if not self.enabled:
            return
        # One line: a line break inside it would end the directive there.
        text = ' '.join(str(text).split())
        if text == self._text:
            return
        self._text = text
        self._text_changed = True
        self._publish(self._reported)

    def finish(self):
        if self.enabled:
            self._publish(1.0)

    def _publish(self, overall):
        self._reported = max(self._reported, min(1.0, max(0.0, overall)))
        now = time.monotonic()
        moved = (self._reported - self._written >= _MIN_STEP
                 and (self._written_at is None or now - self._written_at >= _MIN_INTERVAL))
        finished = self._reported >= 1.0 and self._written < 1.0
        if not (self._text_changed or moved or finished):
            return
        self._written = self._reported
        self._written_at = now
        self._text_changed = False
        line = '@volt:progress %.4f' % self._written
        if self._text:
            line += ' ' + self._text
        sys.stderr.write(line + '\n')
        sys.stderr.flush()


job_progress = _JobProgress()


class ProgressSteps:
    """One weighted split of the current step, used as ``with ProgressSteps(1, 6, 1) as steps:``."""

    def __init__(self, *weights):
        job_progress.begin_steps(weights)

    def next(self):
        job_progress.next_step()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        job_progress.end_steps()
        return False
