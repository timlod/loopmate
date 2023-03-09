from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import KW_ONLY, dataclass, field
from typing import Callable

import numpy as np
import sounddevice as sd
from scipy import signal as sig

from loopmate import config

blend_windowsize = int(config.blend_length * config.sr)
RAMP = np.linspace(1, 0, blend_windowsize, dtype=np.float32)[:, None]


@dataclass
class Action:
    start: int
    end: int
    loop_length: int

    _: KW_ONLY
    # If True, loop this action instead of consuming it
    countdown: int = 0
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
        # don't mix up with current_frame!
        self.current_sample = 0
        self.consumed = False

    def trigger(self, current_frame, next_frame):
        if self.end > self.start:
            return self.start <= current_frame <= self.end
        else:
            # Include case where action lasts all the time, i.e. from 0 to 0
            return (current_frame >= self.end) and (
                current_frame <= self.start
            )

    def index(self, n, current_frame, next_frame):
        offset_a = max(0, self.start - current_frame)
        # If data wraps around we need to index 'past the end'
        offset_b = self.end - current_frame
        if next_frame < current_frame:
            offset_b += next_frame
        return offset_a, offset_b

    def run(self, data):
        self.do(data)
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

    def do(self, outdata):
        raise NotImplementedError("Subclasses need to inherit this!")

    def cancel(self):
        self.current_sample = self.n
        self.loop = False
        self.countdown = 0
        self.consumed = True

    def set_priority(self, priority):
        self.priority = priority


@dataclass
class Trigger:
    when: int
    loop_length: int

    _: KW_ONLY
    # If True, loop this action instead of consuming it
    countdown: int = 0
    # If loop is True, we want to spawn every time
    # else, we want to spawn after countdown and consume
    loop: bool = False
    priority: int = 1
    # As opposed to Action, which spawns when consumed, this class spawns when
    # triggered (such that triggered actions happen this buffer), and does
    # nothing when consumed
    spawn: Action | None = None

    def __post_init__(self):
        self.consumed = False

    def run(self):
        if self.loop:
            pass
        elif self.countdown > 0:
            self.countdown -= 1
        else:
            self.cancel()

    def __lt__(self, other):
        return self.priority < other.priority

    def trigger(self, current_frame, next_frame):
        if current_frame > next_frame:
            if (self.when >= current_frame) or (self.when < next_frame):
                return True
            else:
                return False
        else:
            return current_frame <= self.when < next_frame

    def index(self, current_frame, next_frame):
        if current_frame > next_frame:
            return self.loop_length - current_frame + self.when
        else:
            return self.when - current_frame

    def cancel(self):
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
                  the length of which is set using config.blend_length
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


class Effect(Action):
    def __init__(
        self, start: int, n: int, transformation: Callable, priority: int = 1
    ):
        """Initialize effect which will fade in at a certain frame.

        :param start: start effect at this frame (inside looped audio)
        :param n: length of looped audio
        :param transformation: callable of form f(outdata) which returns an
            ndarray of the same size as outdata
        :param priority: indicate priority at which to queue this action
        """
        super().__init__(0, n, loop=True)
        self.blend = CrossFade()
        self.current_sample = start
        self.transformation = transformation

    def do(self, data):
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


class Start(Action):
    def __init__(self, start: int, priority: int = 100):
        """Initialize effect which will fade in at a certain frame.

        :param start: start effect at this frame (inside looped audio)
        :param priority: indicate priority at which to queue this action
        """
        blend = CrossFade(left_right=False)
        super().__init__(start, start + blend.n, loop=False)
        self.blend = blend

    def do(self, data):
        data_trans = data * 0.0
        data_trans = self.blend(data, data_trans)
        data[:] = data_trans


class Stop(Action):
    def __init__(self, start: int, priority: int = 100):
        """Initialize effect which will fade in at a certain frame.

        :param start: start effect at this frame (inside looped audio)
        :param priority: indicate priority at which to queue this action
        """
        blend = CrossFade()
        super().__init__(start, start + blend.n, loop=False)
        self.blend = blend

    def do(self, data):
        data_trans = data * 0.0
        data_trans = self.blend(data, data_trans)
        data[:] = data_trans


@dataclass
class Actions:
    # keeps and maintains a queue of actions that are fired in the callback
    loop: Loop | Audio
    max: int = 20
    actions: list = field(default_factory=deque)
    active: asyncio.PriorityQueue = field(
        default_factory=asyncio.PriorityQueue
    )
    mute: None | Action = None

    def run(self, outdata, current_frame):
        """Run all actions (to be called once every callback)

        :param outdata: outdata as passed into sd callback (will fill portaudio
            buffer)
        :param current_frame: first sample index of outdata in full audio
        """

        if self.mute is not None:
            self.mute.run(outdata)
            if self.mute.consumed:
                self.mute = None

        # Activate actions (puts them in active queue)
        for action in self.actions:
            if action.start <= current_frame <= action.end:
                self.active.put_nowait(action)
                print(f"putting {action}, {current_frame}")

        for i in range(self.active.qsize()):
            action = self.active.get_nowait()
            # Note that this only gets triggered if start is within or after
            # the current frame, that's why we can use max
            offset_a = max(0, action.start - current_frame)
            # Indexing past the end of outdata will just return the full array
            offset_b = action.end - current_frame
            action.run(outdata[offset_a:offset_b])
            if action.consumed:
                self.actions.remove(action)
                if isinstance(action, Stop):
                    outdata[offset_b:] = 0.0
                    while not self.active.empty():
                        try:
                            self.active.get(False)
                        except Exception:
                            continue
                    raise sd.CallbackStop()
                if action.spawn is not None:
                    print(f"Putting {action.spawn}")
                    self.actions.append(action.spawn)
