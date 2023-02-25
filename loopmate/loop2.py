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
import soundcard as sc
import sounddevice as sd
import soundfile as sf
import typer
from scipy import signal as sig


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


device = "default"
latency = 0.002
blocksize = 64

# Precompute windows for pop protection
windowsize = min(blocksize, 128)
windows = {
    i: sig.windows.hann(
        i * 2,
    )[:, None]
    for i in range(windowsize + 1)
}


def no_pop(buffer, pos=0, up=True):
    n = len(buffer)
    length = min(windowsize, n - pos if up else pos)
    if up:
        buffer[pos : pos + length] *= windows[length][:length]
    else:
        buffer[pos - length : pos] *= windows[length][-length:]


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


class Fades(Enum):
    # Maybe create fade windows in advance for faster speed
    pass


@dataclass
class Multiplier(Action):
    """
    Simple action which multiplies incoming data by some array.  arr needs to
    be broadcastable such that data * arr works.
    """

    arr: np.ndarray[np.float32]

    def __post_init__(self):
        super().__post_init__()
        assert (
            len(self.arr) == self.n
        ), "Supplied array needs to have length n (end - start)!"
        if len(self.arr.shape) == 1:
            self.arr = self.arr[:, None]

    def do(self, data):
        n_i = len(data)
        data[:] *= self.arr[self.current_sample : self.current_sample + n_i]
        self.current_sample += n_i


# TODO think about pre-made fade for loop borders
@dataclass
class Fade(Action):
    # TODO: fades should also be required by many other actions as a multiplier
    # with the effect rather than outdata itself - can we somehow merge this?
    out: bool

    _: KW_ONLY
    priority: int = 1
    # Use to trigger mute after fade out, or signal mute before fade-in?
    # TODO: Just specify spawn
    # mute: bool = False
    # stop: bool = False

    def __post_init__(self):
        super().__post_init__()
        window = sig.windows.hann(self.n * 2)
        if self.out:
            self.window = window[self.n :, None]
        else:
            self.window = window[: self.n, None]

    def do(self, data):
        n_i = len(data)
        data[:] *= self.window[self.current_sample : self.current_sample + n_i]
        self.current_sample += n_i


class Mute(Multiplier):
    def __init__(self, n):
        silence = np.zeros(n, np.float32)
        super().__init__(0, n, silence, recurring=True, priority=0)

    @classmethod
    def from_fadeout(cls, fade_end: int, length: int):
        """Intermediate mute spawned from fadeout, which in turn spawns a full
        loop mute.

        :param fade:
        :param length:
        """
        n = length - fade_end
        silence = np.zeros(n)
        spawn = Mute(length)
        return Multiplier(
            fade_end,
            fade_end + n,
            silence,
            priority=0,
            recurring=False,
            spawn=spawn,
        )

    def unmute(self, current_frame, start=None, left=True):
        if start is None:
            self.current_sample = self.n
        else:
            self.current_sample = start - current_frame

        self.recurring = False
        self.spawn = Fade(start, start + windowsize, priority=0)


@dataclass
class Stop(Action):
    start: int = 0
    end: int = 0
    recurring: bool = False

    def do(self, outdata):
        outdata[:] = 0.0
        print("In Stop!")
        raise sd.CallbackAbort()


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
            print(f"{action} on queue")
            if action.start <= current_frame <= action.end:
                self.active.put_nowait(action)

        for i in range(self.active.qsize()):
            action = self.active.get_nowait()
            print(f"Performing {action}")
            # Note that this only gets triggered if start is within or after
            # the current frame, that's why we can use max
            offset_a = max(0, action.start - current_frame)
            # Indexing past the end of outdata will just return the full array
            offset_b = action.end - current_frame
            action.run(outdata[offset_a:offset_b])
            if not action.consumed:
                self.q.put_nowait(action)
            else:
                if action.spawn is not None:
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
        pop_window = self.sr * self.pop_window_ms
        window = sig.windows.hann(pop_window)
        self.audio[: pop_window // 2] *= window[: pop_window // 2]
        if pop_window > 0:
            self.audio[-(pop_window // 2) :] *= window[-(pop_window // 2) :]

        self.n = len(self.audio)
        self.n_frames = np.ceil(self.n / blocksize)
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            device=device,
            channels=self.audio.shape[1],
            callback=self._get_callback(),
            latency=latency,
            blocksize=blocksize,
        )
        self.trans_left = False
        self.trans_right = False
        self.tasks = asyncio.Queue()
        self.raudio = self.audio[::-1]
        self._current_frame_i = self.current_frame // self.blocksize

    def reverse(self):
        self._reverse = not self._reserve

    async def trigger(self, x):
        pass

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
            print(time.currentTime - time.outputBufferDacTime)
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
            # To loop, add start of audio to end of output buffer:
            if leftover <= frames:
                outdata[chunksize:] = self.audio[: frames - leftover]
                self.current_frame = frames - leftover
                self._current_frame_i = 0
            else:
                self.current_frame += frames
                self._current_frame_i += 1

            # Transformations:
            outdata[:] = self.transformation(self.gain * outdata)

            # Loop over trigger list to see if we should schedule a task
            # There should be a task queue to empty inside the next callback
            removes = []
            for trigger in self.triggers:
                if trigger.sample < next_frame:
                    if not trigger.recurrent:
                        removes.append(trigger)
            for trigger in removes:
                self.triggers.remove(trigger)

        return callback

    async def start(self):
        print("starting stream")
        self.trans_left = True
        self.stream.start()

    async def stop(self, reset=True):
        print("stopping stream")
        if reset:
            self.current_frame = 0
        self.trans_right = True
        self.stream.stop()


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
                loop.tasks.put(loop.stop_next)
            if c == "q":
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
