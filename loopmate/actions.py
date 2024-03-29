"""
actions
=======

Actions and Triggers which operate on audio loops in realtime.

Defines 3 important classes:

    1. Actions

    - a class containing deques/priority queues contains Action objects which
      operate on stretches of looped audio.  It contains logic to fire triggers
      and run actions on audio buffers in real-time.

    2. Action:

    - A superclass that defines functionality to seamlessly apply some
      operations to audio buffers in real-time.  Examples: Mute, Play sample or
      apply Audio FX

    3. Trigger:

    - A superclass that defines functionality to trigger some action when a
      certain point in looped audio is reached.  Examples: trigger Mute or
      Record actions when the loop boundary is reached (on the next loop)
"""

from __future__ import annotations

import queue
from collections import deque
from dataclasses import KW_ONLY, dataclass, field
from typing import Callable

import numpy as np
import sounddevice as sd

from loopmate import config

RAMP = np.linspace(1, 0, config.BLEND_SAMPLES, dtype=np.float32)[:, None]


@dataclass
class Action:
    """
    Action that manipulates audio buffers at specified times.
    """

    start: int
    end: int
    loop_length: int

    _: KW_ONLY
    countdown: int = 0
    # If True, loop this action instead of consuming it
    loop: bool = False
    priority: int = 3
    # Consuming this action will 'spawn'/queue this new action
    spawn: Action | None = None

    def __post_init__(self):
        if self.end > self.start:
            self.n = self.end - self.start
        else:
            self.n = self.start + self.loop_length - self.end

        # Current sample !inside action between start and end
        # don't mix up with current_index in loop!
        self.current_sample = 0
        self.consumed = False

    def trigger(self, current_index, next_index):
        """Run at every step to signal when the out buffer enters the start/end
        boundaries of this action.

        :param current_index: first sample index of outdata in full audio loop
        :param next_index: first sample index of outdata in full audio loop for
            the next step.  Will be != current_index + n_samples only when
            wrapping around at the loop boundary.
        """
        if self.end > self.start:
            return self.start <= current_index <= self.end
        else:
            # Include case where action lasts all the time, i.e. from 0 to 0
            return (current_index >= self.end) and (
                current_index <= self.start
            )

    def index_outdata(self, current_index, next_index):
        """Return offsets for indexing part of the outdata buffer to use in
        this action's next run step.  This is mostly to accomodate edge cases
        where an action starts/ends within a buffer.

        :param current_index: first sample index of outdata in full audio loop
        :param next_index: first sample index of outdata in full audio loop for
            the next step.  Will be != current_index + n_samples only when
            wrapping around at the loop boundary.
        """
        # Note that this only gets triggered if start is within or after
        # the current block, that's why we can use max
        offset_a = max(0, self.start - current_index)
        # If data wraps around we need to index 'past the end'
        offset_b = self.end - current_index
        if next_index < current_index:
            offset_b += next_index
        return offset_a, offset_b

    def run(self, data, current_index):
        """Run action on incoming audio and updates internal counters
        accordingly.

        :param data: audio buffer
        """
        self.do(data, current_index)
        self.current_sample += len(data)

        if self.current_sample >= self.n:
            if self.loop:
                self.current_sample = 0
            elif self.countdown > 0:
                self.current_sample = 0
                self.countdown -= 1
            else:
                self.consumed = True

    def __lt__(self, other):
        return self.priority < other.priority

    def do(self, outdata, current_index):
        """Perform manipulations on the output buffer.

        :param outdata: output buffer
        """
        raise NotImplementedError("Subclasses need to override this!")

    def cancel(self):
        """Immediately cancel/stop this action."""
        self.current_sample = self.n
        self.loop = False
        self.countdown = 0
        self.consumed = True

    def set_priority(self, priority):
        self.priority = priority


class CrossFade:
    def __init__(self, n=None, left_right=True):
        """Initialize blending operation across multiple audio buffers.
        Blending is performed as a simple linear interpolation.

        Call using CrossFade()(a, b).

        :param n: length of the blending (in samples).  By default use RAMP,
                  the length of which is set using config.BLEND_LENGTH
        :param left_right: if False, b blends into a instead of a -> b
        """
        if n is None:
            self.ramp = RAMP
            n = len(self.ramp)
        else:
            self.ramp = np.linspace(1, 0, n, dtype=np.float32)[:, None]
        self.left_right = left_right
        self.n = n
        self.i = 0
        self.done = False

    def __call__(self, a: np.ndarray, b: np.ndarray):
        """Blend a and b. Direction is controlled by self.left_right.

        Call this once per callback, and stop applying once self.done.

        :param a: array to blend with b
        :param b: array to blend with a
        """
        ramp = self.ramp[self.i : self.i + len(a)]
        n = len(ramp)
        if self.left_right:
            out = b
            out[:n] = ramp * a[:n] + (1 - ramp) * b[:n]
        else:
            out = a
            out[:n] = ramp * b[:n] + (1 - ramp) * a[:n]
        self.i += n
        if self.i == self.n:
            self.done = True
        return out


# TODO:
# Traceback (most recent call last):
#   File "/home/tim/projects/loopmate/loopmate/main.py", line 71, in receive
#     self.command(note)
#   File "/home/tim/projects/loopmate/loopmate/main.py", line 131, in command
#     self.loop.measure_air_delay()
#   File "/home/tim/projects/loopmate/loopmate/loop.py", line 425, in measure_air_delay
#     rec_audio = self.rec_audio[-samples_waited + indelay_samples :]
#                 ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/tim/projects/loopmate/loopmate/circular_array.py", line 106, in __getitem__
#     return self.query(i)
#            ^^^^^^^^^^^^^
#   File "/home/tim/projects/loopmate/loopmate/circular_array.py", line 97, in query
#     return query_circular(
#            ^^^^^^^^^^^^^^^
#   File "/home/tim/projects/loopmate/loopmate/circular_array.py", line 31, in query_circular
#     -N <= start < stop <= 0
# AssertionError: Can only slice at most N (2646000) items backward on! 455/0


class Sample(Action):
    def __init__(
        self,
        sample: np.ndarray,
        loop_length: int,
        wait: int = 0,
        gain: float = 1.0,
    ):
        """(Immediately) play this sample when put into actions.  This is
        currently kind of hacky as it overwrites essential functionality of the
        Action class to allow triggering without an anchor present.

        :param sample: array containing sample to play
        :param loop_length: length of containing loop, to make sure that we can
            immediately start playing
        :param wait: if playback at an exact time is needed (over
            as-quick-as-possible playback), start this many samples into the
            first data array
        :param gain: gain to apply to sample
        """
        super().__init__(0, loop_length, loop_length)
        self.sample = sample
        # Overwrite n from Action (which would be set to loop_length) to allow
        # playing samples longer than loop_length
        self.n = len(sample)
        self.wait = wait
        self.gain = gain

    def do(self, data, current_index):
        sample = self.sample[
            self.current_sample : self.current_sample + len(data) - self.wait
        ]
        data[self.wait : len(sample) + self.wait] += self.gain * sample
        # We only wait once in the beginning, if at all
        if self.wait:
            self.current_sample -= self.wait
            self.wait = 0

    def index(self, current_index, next_index):
        return 0, self.n


class Effect(Action):
    def __init__(
        self, start: int, n: int, transformation: Callable, priority: int = 1
    ):
        """Initialize effect which will fade in starting at a certain sample.

        :param start: start effect at this sample (inside looped audio)
        :param n: length of looped audio
        :param transformation: callable of form f(outdata) which returns an
            ndarray of the same size as outdata
        :param priority: indicate priority at which to queue this action
        """
        super().__init__(0, n, n, loop=True)
        self.blend = CrossFade()
        self.current_sample = start
        self.transformation = transformation

    def do(self, data, current_index):
        """Called from within run inside callback. Applies the effect.

        :param data: outdata in callback, modified in-place
        """
        data_trans = self.transformation(data)
        if self.blend is not None:
            data_trans = self.blend(data, data_trans)
            if self.blend.done:
                self.blend = None
        data[:] = data_trans

    def cancel(self):
        """Stops effect over the next buffer(s).  Fades out to avoid audio
        popping, and may thus take several callbacks to complete.
        """
        self.blend = CrossFade(left_right=False)
        self.current_sample = self.n - self.blend.n
        self.loop = False


class Mute(Effect):
    def __init__(self, start: int, n: int):
        super().__init__(start, n, lambda x: x * 0.0, priority=0)


class Start(Action):
    def __init__(self, start: int, loop_length: int, priority: int = 100):
        """Initialize Start action which will fade in containing audio.

        :param start: start effect at this sample (inside looped audio)
        :param priority: indicate priority at which to queue this action
        """
        blend = CrossFade(left_right=False)
        super().__init__(start, start + blend.n, loop_length, loop=False)
        self.blend = blend

    def do(self, data, current_index):
        data_trans = data * 0.0
        data_trans = self.blend(data, data_trans)
        data[:] = data_trans


class Stop(Action):
    def __init__(self, start: int, loop_length: int, priority: int = 100):
        """Initialize Stop action which will fade out containing audio.

        :param start: start effect at this sample (inside looped audio)
        :param priority: indicate priority at which to queue this action
        """
        blend = CrossFade()
        super().__init__(start, start + blend.n, loop_length, loop=False)
        self.blend = blend

    def do(self, data, current_index):
        data_trans = data * 0.0
        data_trans = self.blend(data, data_trans)
        data[:] = data_trans


@dataclass
class Trigger:
    """
    Trigger that fires when a certain location in the loop is reached.
    """

    when: int
    loop_length: int

    _: KW_ONLY
    # If True, loop this action instead of consuming it until the countdown
    # reaches 0.
    countdown: int = 0
    # If loop is True, we want to spawn every time
    # else, we want to spawn after countdown and consume
    loop: bool = False
    priority: int = 1
    # As opposed to Action, which spawns when consumed, this class spawns when
    # triggered (such that triggered actions happen this buffer). If loop is
    # True, consumption and countdown are reset after triggering
    spawn: Action | None = None

    def __post_init__(self):
        self.consumed = False
        self.i = self.countdown

    def run(self, actions, current_index):
        """Run this trigger.  This means count down to trigger or consume.

        :param actions: Actions object that can be manipulated if triggered
        """
        if self.i > 0:
            self.i -= 1
            self.consumed = False
        else:
            self.do(actions, current_index)
            if self.loop:
                self.i = self.countdown
            else:
                self.consumed = True

    def do(self, actions, current_index):
        """By default, simply put this trigger in the plan queue.  Override to
        trigger different things.

        :param actions: Actions object
        """
        actions.plans.put_nowait(self)

    def __lt__(self, other):
        return self.priority < other.priority

    def trigger(self, current_index, next_index):
        """Returns True if this trigger activates given the current location.

        :param current_index: first sample index of outdata in full audio loop
        :param next_index: first sample index of outdata in full audio loop for
            the next step.  Will be != current_index + n_samples only when
            wrapping around at the loop boundary.
        """
        if current_index > next_index:
            if (self.when >= current_index) or (self.when < next_index):
                return True
            else:
                return False
        else:
            return current_index <= self.when < next_index

    def index(self, current_index, next_index):
        """Return offsets for indexing part of the outdata buffer to use in
        this triggers's next run step.

        :param current_index: first sample index of outdata in full audio loop
        :param next_index: first sample index of outdata in full audio loop for
            the next step.  Will be != current_index + n_samples only when
            wrapping around at the loop boundary.
        """
        if current_index > next_index:
            return self.loop_length - current_index + self.when
        else:
            return self.when - current_index

    def cancel(self):
        self.loop = False
        self.countdown = 0
        self.consumed = True

    def set_priority(self, priority):
        self.priority = priority


class MuteTrigger(Trigger):
    def __init__(self, when, loop_length, **kwargs):
        super().__init__(when, loop_length, **kwargs)

    def do(self, actions, current_index):
        if len(actions.actions) > 0 and isinstance(actions.actions[0], Mute):
            actions.actions[0].cancel()
        else:
            mute = Mute(self.when, self.loop_length)
            actions.actions.appendleft(mute)
            actions.active.put_nowait(mute)


class RecordTrigger(Trigger):
    def __init__(self, when, loop_length, **kwargs):
        super().__init__(when, loop_length, **kwargs)


class BackCaptureTrigger(Trigger):
    def __init__(self, when, loop_length, n_loops=1, **kwargs):
        super().__init__(when, loop_length, **kwargs)
        self.n_loops = n_loops


@dataclass
class Actions:
    # keeps and maintains a queue of actions that are fired in the callback
    max: int = 20
    actions: deque = field(default_factory=deque)
    active: queue.PriorityQueue = field(default_factory=queue.PriorityQueue)
    plans: queue.PriorityQueue = field(default_factory=queue.PriorityQueue)

    def append(self, action: Action | Trigger):
        self.actions.append(action)

    def prepend(self, action: Action | Trigger):
        self.actions.insert(0, action)

    def run(self, outdata, current_index, next_index):
        """Run all actions (to be called once every callback)

        :param outdata: outdata as passed into sd callback (will fill portaudio
            buffer)
        :param current_index: first sample index of outdata in full audio loop
        :param next_index: first sample index of outdata in full audio loop for
            the next step.  Will be != current_index + n_samples only when
            wrapping around at the loop boundary.
        """
        # Activate actions (puts them in active queue)
        for action in self.actions:
            if action.trigger(current_index, next_index):
                self.active.put_nowait(action)

        while not self.active.empty():
            action = self.active.get_nowait()
            if isinstance(action, Trigger):
                print(f"Trigger {action}, {current_index}")
                action.run(self, current_index)
                if action.consumed:
                    print(self.plans)
                    if not action.loop:
                        self.actions.remove(action)
                    if action.spawn is not None:
                        self.actions.append(action.spawn)
                continue

            # Actions
            offset_a, offset_b = action.index_outdata(
                current_index, next_index
            )
            action.run(outdata[offset_a:offset_b], current_index)
            if action.consumed:
                print(f"consumed {action}")
                self.actions.remove(action)
                if isinstance(action, Stop):
                    outdata[offset_b:] = 0.0
                    while not self.active.empty():
                        try:
                            self.active.get(False)
                        except Exception:
                            break
                    raise sd.CallbackStop()
                if action.spawn is not None:
                    print(f"Spawning {action.spawn}")
                    self.actions.append(action.spawn)
