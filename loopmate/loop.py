from __future__ import annotations

import asyncio
import queue
import sys
import termios
import threading
import time
import tty
from dataclasses import KW_ONLY, dataclass
from enum import Enum, member
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd
import pedalboard
import soundcard as sc
import sounddevice as sd
import soundfile as sf
import typer
from scipy import signal as sig

from loopmate import config

# TODO:
#
# - wrapper class around tasks that handles e.g. a full mute, with fade in/mute
# - don't use action queue the way it currently works, or add some outer list
# - that holds current tasks such that they can be selected
# - allow tasks to be cancelled immediately / in the next callback


def chirp(
    f0, f1, l=10, a=1.0, method: str = "logarithmic", phi=-90, sr: int = 44100
):
    """
    Generates a sine sweep between two frequencies of a given length.
    :param f0: Frequency (in Hertz) at t=0
    :param f1: Frequency (in Hertz) at t=l*sr (end frequency of the sweep)
    :param l: Length (in seconds) of the sweep
    :param a: Amplitude scaling factor (default=1), can be array-like of
              length l * sr to provide element-wise scaling.
    :param method: linear, logarithmic, quadratic or hyperbolic
    :param phi: Phase offset in degrees (default=-90 - to start at 0 amplitude)
    :param sr: Sampling rate of the generated waveform (default=44100 Hz)
    :return: Array containing the sweep
    """
    t = np.linspace(0, l, int(l * sr), endpoint=False)
    return a * sig.chirp(t, f0, l, f1, method=method, phi=phi)


blend_windowsize = int(config.blend_length * config.sr)
RAMP = np.linspace(1, 0, blend_windowsize, dtype=np.float32)[:, None]


def blend(x, n=None):
    if n is None:
        ramp = np.linspace(1, 0, n)
    else:
        ramp = RAMP
    return ramp + x - ramp * x


def blend2(x1, x2, n=None):
    if n is None:
        ramp = np.linspace(1, 0, n)
    else:
        ramp = RAMP
    return ramp * x1 + (1 - ramp) * x2


# A mute could just be a 0 multiplier that is blended in/out
# Make sure the blend is shorter than the audio
# Every configuration change has to trigger a blend action


@dataclass
class Action:
    start: int
    end: int

    _: KW_ONLY
    # If True, loop this action instead of consuming it
    loop: bool = False
    priority: int = 3
    # Consuming this action will 'spawn'/queue this new action
    spawn: Action | None = None

    def __post_init__(self):
        self.n = self.end - self.start
        # Current sample !inside action between start and end
        # don't mix up with current_frame!
        self.current_sample = 0
        self.consumed = False

    def run(self, data):
        self.do(data)
        self.current_sample += len(data)

        if self.current_sample >= self.n:
            if self.loop:
                self.current_sample = 0
            else:
                self.consumed = True

    def __lt__(self, other):
        return self.priority < other.priority

    def do(self, outdata):
        raise NotImplementedError("Subclasses need to inherit this!")

    def cancel(self):
        # TODO: initiate fade-out
        pass

    def set_priority(self, priority):
        self.priority = priority


class Blender:
    def __init__(self, n=None, left_right=True):
        """Initialize blending operation across multiple audio buffers.
        Blending is performed as a simple linear interpolation.

        Call using Blender()(a, b).

        :param n: length of the blending (in samples).  By default use RAMP,
                  the length of which is set using config.blend_length
        :param left_right: if False, b blends into a instead of a -> b
        """
        if n is None:
            self.ramp = RAMP
            n = len(self.ramp)
        else:
            self.ramp = np.linspace(1, 0, n, dtype=np.float32)
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
        self.blend = Blender()
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

    def stop(self):
        """Stops effect over the next buffer(s).  Fades out to avoid audio
        popping, and may thus take several callbacks to complete.
        """
        self.blend = Blender(left_right=False)
        self.current_sample = self.n - self.blend.n
        self.loop = False


@dataclass
class Actions:
    # keeps and maintains a queue of actions that are fired in the callback
    loop: Loop
    max: int = 20
    q = asyncio.PriorityQueue(maxsize=max)
    active = asyncio.PriorityQueue(maxsize=max)
    # TODO: if true, reset frames in all actions
    stop: bool = False

    def run(self, outdata, current_frame):
        """Run all actions (to be called once every callback)

        :param outdata: outdata as passed into sd callback (will fill portaudio
            buffer)
        :param current_frame: first sample index of outdata in full audio
        """
        if self.stop:
            # Add fade out + stop to active queue
            self.stop = False
            self.q.put_nowait(
                Fade(
                    current_frame,
                    current_frame + 1024,
                    out=True,
                    priority=0,
                    spawn=Stop(current_frame + 1024, current_frame + 1024),
                )
            )

        # Activate actions (puts them in active queue)
        for i in range(self.q.qsize()):
            action = self.q.get_nowait()
            if action.start <= current_frame <= action.end:
                self.active.put_nowait(action)

        for i in range(self.active.qsize()):
            action = self.active.get_nowait()
            # print(f"Performing {action}")
            # Note that this only gets triggered if start is within or after
            # the current frame, that's why we can use max
            offset_a = max(0, action.start - current_frame)
            # Indexing past the end of outdata will just return the full array
            offset_b = action.end - current_frame
            action.run(outdata[offset_a:offset_b])
            if any(outdata > 0):
                print(
                    action,
                    current_frame,
                    action.current_sample,
                    action.end,
                    offset_a,
                    offset_b,
                    offset_b - offset_a,
                )
            if not action.consumed:
                self.q.put_nowait(action)
            else:
                if action.spawn is not None:
                    print(f"Putting {action.spawn}")
                    self.q.put_nowait(action.spawn)


@dataclass
class Loop:
    audio: np.ndarray
    gain: float = 1.0
    mute: bool = False
    transformation: callable = lambda x: x
    _reverse: bool = False
    current_frame: int = 0
    sr: int = 96000
    pop_window_ms: float = 0.005

    def __post_init__(self):
        # Remove pop at loop edge
        pop_window = int(self.sr * self.pop_window_ms)
        window = sig.windows.hann(pop_window)[:, None]
        self.audio[: pop_window // 2] *= window[: pop_window // 2]
        if pop_window > 0:
            self.audio[-(pop_window // 2) :] *= window[-(pop_window // 2) :]

        self.n = len(self.audio)
        self.n_frames = np.ceil(self.n / config.blocksize)
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            device=config.device,
            channels=self.audio.shape[1],
            callback=self._get_callback(),
            latency=config.latency,
            blocksize=config.blocksize,
        )
        self.trans_left = False
        self.trans_right = False
        self.raudio = self.audio[::-1]
        self._current_frame_i = self.current_frame // config.blocksize

        self.actions = Actions(self)

    def _get_callback(self):
        """
        Creates callback function for this loop.
        """

        def callback(outdata, frames, time, status):
            if status:
                print(status)

            leftover = self.n - self.current_frame
            chunksize = min(leftover, frames)
            next_frame = self.current_frame + chunksize

            # print(f"playing at {self.current_frame}")
            # print(time.currentTime - time.outputBufferDacTime)

            outdata[:chunksize] = self.audio[
                self.current_frame : self.current_frame + chunksize
            ]
            if leftover <= frames:
                outdata[chunksize:] = self.audio[: frames - leftover]

            self.actions.run(outdata, self.current_frame)
            # To loop, add start of audio to end of output buffer:
            if leftover <= frames:
                self.current_frame = frames - leftover
                self._current_frame_i = 0
            else:
                self.current_frame += frames
                self._current_frame_i += 1

        return callback

    async def start(self):
        print("starting stream")
        self.current_frame = 0
        self.stream.start()

    async def stop(self, reset=True):
        print("stopping stream")
        # if reset:
        #     self.current_frame = 0
        # self.trans_right = True
        # self.stream.stop()
        self.actions.stop = True


class CharIn:
    def __init__(self, loop=None):
        self.loop = loop or asyncio.get_event_loop()
        self.q = asyncio.Queue()
        self.loop.add_reader(sys.stdin, self.got_input)

    def got_input(self):
        asyncio.ensure_future(self.q.put(sys.stdin.read(1)), loop=self.loop)

    async def __call__(self):
        return await self.q.get()


async def main():
    print(sd.query_devices())
    sr = 96000
    sweep = 0.2 * chirp(200, 1000, 2, method="linear", sr=sr)
    loop = Loop(sweep[:, None])
    print(loop)

    try:
        prompt = CharIn()
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        while True:
            c = await prompt()
            print(c, end="\n\r")
            if c == "s":
                await loop.start()
            if c == "m":
                loop.mute = not loop.mute
                print(loop.mute)
                if loop.mute:
                    print("MUTING")
                    loop.actions.q.put_nowait(
                        Effect(loop.current_frame, loop.n, lambda x: x * 0.0)
                    )
                else:
                    mute = loop.actions.q._queue[0]
                    mute.stop(loop.current_frame)
            if c == "o":
                await loop.stop()
            if c == "r":
                loop.reverse = True
            if c == "a":
                loop.gain = 0.5
            if c == "b":
                loop.gain = 1.0
            if c == "c":
                # schedule stop task at end of loop
                pass
                # loop.tasks.put(loop.stop_next)
            if c == "q":
                break
    except (sd.CallbackStop, sd.CallbackAbort):
        print("Stopped")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
