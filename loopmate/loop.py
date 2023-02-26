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
RAMP = np.linspace(1, 0, blend_windowsize)


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


@dataclass
class Action:
    start: int
    end: int

    _: KW_ONLY
    recurring: bool = False
    priority: int = 3
    # Consuming this action will 'spawn'/queue this new action
    spawn: Action | None = None

    def __post_init__(self):
        self.n = self.end - self.start
        # Current sample !inside action between start and end
        # don't mix up with current_frame!
        self.current_sample = 0
        self.consumed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_sample >= self.n:
            if self.recurring:
                self.current_sample = 0
            raise StopIteration
        return self.do

    def run(self, data):
        self.do(data)
        if self.recurring:
            self.current_sample = 0

        if self.current_sample >= self.n:
            self.consumed = True

    def __lt__(self, other):
        return self.priority < other.priority

    def do(self, outdata):
        raise NotImplementedError("Subclasses need to inherit this!")

    def set_priority(self, priority):
        self.priority = priority


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
            # TODO: move into action after buffer is filled
            # if self.mute:
            #     outdata[:] = 0.0
            #     if leftover <= frames:
            #         self.current_frame = frames - leftover
            #     else:
            #         self.current_frame += frames
            #     return

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
                        Mute3(loop.current_frame, loop.n)
                    )
                else:
                    mute = loop.actions.q.get_nowait()
                    mute.unmute(loop.current_frame)
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
