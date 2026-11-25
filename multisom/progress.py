"""How far the run is, as VOLT shows it on the job.

Mirrors coretoolkit's ``Volt::JobProgress``: ``@volt:progress <0..1> <text>``
lines on STDERR, composed from weighted steps that may split again (the way
OVITO composes its sub-steps), so the value climbs once across every pass and
never goes back. Nothing is written unless VOLT sets ``VOLT_PROGRESS=1``.
``Progress`` mirrors ``Volt::ProgressReporter`` for a long loop.

The text is what the current step does: a pass's label while it runs, or the
one given to ``next_step``. When a step is over its text goes with it, back to
the one of the step that was split. A run that fails stops where it was; only
``finish()`` ends it at 1.

A persistent host runs many frames in one process, so ``start()`` begins each run
from zero.
"""

import os
import sys
import time

_MIN_STEP = 0.005
_MIN_INTERVAL = 0.2
_NEVER = float('inf')


class _Level:
    __slots__ = ('base', 'span', 'bounds', 'index', 'text')

    def __init__(self, base, span, bounds, text):
        self.base = base
        self.span = span
        self.bounds = bounds
        self.index = 0
        # The text of the step this split divides.
        self.text = text


def _one_line(text):
    # A line break inside the text would end the directive there.
    return ' '.join(str(text).split())


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
        for level in self._levels:
            low = level.base + level.span * level.bounds[level.index]
            high = level.base + level.span * level.bounds[level.index + 1]
        return low, high

    def begin_steps(self, weights, text=''):
        """Splits the current step into steps of these relative weights and enters the first, doing ``text``."""
        if not self.enabled:
            return
        low, high = self._range()
        clean = [max(0.0, float(weight)) for weight in weights] or [1.0]
        total = sum(clean)
        bounds, cumulative = [0.0], 0.0
        for weight in clean:
            cumulative += weight
            bounds.append(cumulative / total if total > 0 else 1.0)
        self._levels.append(_Level(low, high - low, bounds, self._text))
        self._show_step(text)

    def next_step(self, text=''):
        """The current step is done and the next one of the split starts, doing ``text``."""
        if not self.enabled or not self._levels:
            return
        level = self._levels[-1]
        if level.index + 2 < len(level.bounds):
            level.index += 1
        self._show_step(text)

    def end_steps(self):
        """The split is over; the step it divided goes on, so the value stays where it is."""
        if not self.enabled or not self._levels:
            return
        self._restore_text(self._levels.pop().text)
        self._publish(self._reported)

    def set_fraction(self, fraction):
        if not self.enabled:
            return
        low, high = self._range()
        self._publish(low + min(1.0, max(0.0, float(fraction))) * (high - low))

    def set_text(self, text):
        """What the run is doing now, until the step it was set in is over."""
        if not self.enabled:
            return
        self._restore_text(_one_line(text))
        self._publish(self._reported)

    def finish(self):
        if self.enabled:
            self._publish(1.0)

    def _show_step(self, text):
        # A step with no text is described by the step its split divides, and a step of no weight (a phase
        # this run skips) keeps the text it found, since nothing runs in it.
        low, high = self._range()
        if high > low:
            self._restore_text(_one_line(text) if text else self._levels[-1].text)
        self._publish(low)

    def _restore_text(self, text):
        if text != self._text:
            self._text = text
            # VOLT keeps the last text it was given, so no text is no news.
            self._text_changed = self._text_changed or bool(text)

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
    """One weighted split of the current step, used as::

        with ProgressSteps(('Reading the input', 1), ('Training', 6), ('Writing outputs', 1, writes)) as steps:
            ...; steps.next(); ...

    Each step is ``(text, weight)``, or ``(text, weight, runs)`` for a step this run may skip.
    """

    def __init__(self, *steps):
        self._texts = [step[0] for step in steps]
        self._current = 0
        weights = [step[1] if len(step) < 3 or step[2] else 0.0 for step in steps]
        job_progress.begin_steps(weights, self._texts[0] if self._texts else '')

    def next(self):
        """The current step is done; the next one starts."""
        if self._current + 1 < len(self._texts):
            self._current += 1
            job_progress.next_step(self._texts[self._current])

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        job_progress.end_steps()
        return False


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
        self._last_line = time.monotonic()
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
        print('%s: %d%% (%d/%d%s)' % (self.label, done * 100 // self.total, done, self.total, self._suffix),
              file=sys.stderr, flush=True)
