import asyncio
import queue
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
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


def make_loop_callback(data: np.ndarray):
    """TODO

    Can probably add audio transformations in here as well, and interpolate
    based on some temparature and the current time in the loop/session.

    Every loop should be able to change state.

    TODO: MUTE, RESTART (on trigger)

    :param data: audio data to loop

    :returns:
    """

    n = len(data)
    current_frame = 0

    def callback(outdata, frames, time, status):
        if status:
            print(status)

        leftover = n - current_frame
        chunksize = min(leftover, frames)
        outdata[:chunksize] = data[current_frame : current_frame + chunksize]
        # to loop:
        if leftover < frames:
            outdata[chunksize:] = data[: frames - leftover]
            current_frame = frames - leftover
        else:
            current_frame += chunksize

    return callback


device = "default"
latency = 0.002
blocksize = 256

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


class Tasks(Enum):
    @member
    def p():
        print("Hi")

    @member
    def a():
        print("a")


class Task:
    # Something that can modify streams/callbacks within streams
    pass


@dataclass
class TimedTask:
    # Work in samples or seconds? probably samples is better
    start: float
    end: float
    sr: int
    task: Tasks
    # Should


async def worker(loop, q):
    # Perhaps give each task a time/event when it should be triggered, then use
    # asyncio to actually execute once that's the case? E.g., for reverse, we
    # have to reverse the audio before we actually restart the loop, otherwise
    # it will probably reverse on the second frame. For restart, we may also
    # want to trigger it when a timer counts down, and then put it in the
    # queue. Ergo, we need timekeeping somehow.
    while True:
        task = await q.get()
        # Do task
        if task == "start":
            loop.start()
        elif task == "stop":
            loop.stop()
        elif task == "reverse":
            pass
        q.task_done()


@dataclass
class Loop:
    audio: np.ndarray
    gain: float = 1.0
    mute: bool = False
    transformation: callable = lambda x: x
    _reverse: bool = False
    current_frame: int = 0
    sr: int = 96000

    def __post_init__(self):
        self.n = len(self.audio)
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

    def reverse(self):
        self.audio = self.audio[::-1]
        self._reverse = not self._reserve

    async def loop_start(self, outdata, start):
        """Empties task queue when

        :param outdata:
        :param start:
        :returns:

        """
        # This should be called whenever the loop is reset; the task queue will
        # be emptied

        # Reduce pop - perhaps don't do this here, and just empty the task
        # queue
        no_pop(outdata, start, up=True)
        task = asyncio.create_task(worker(self.__repr__(), self.tasks))
        await self.tasks.join()
        task.cancel()
        await task

    def _get_callback(self):
        """
        Creates callback function for this loop.
        """

        def callback(outdata, frames, time, status):
            if status:
                print(status)
            # if self.stop:
            #     self.stop = False
            #     raise sd.CallbackStop

            leftover = self.n - self.current_frame

            print(f"playing at {self.current_frame}")
            if self.mute:
                outdata[:] = 0.0
                if leftover <= frames:
                    self.current_frame = frames - leftover
                else:
                    self.current_frame += frames
                return

            chunksize = min(leftover, frames)

            outdata[:chunksize] = self.audio[
                self.current_frame : self.current_frame + chunksize
            ]
            if self.trans_left:
                no_pop(outdata, 0, True)
                self.trans_left = False
            if self.trans_right:
                no_pop(outdata, chunksize - windowsize, False)
                self.trans_right = False
            # To loop, add start of audio to end of output buffer:
            if leftover <= frames:
                outdata[chunksize:] = self.audio[: frames - leftover]
                no_pop(outdata, chunksize, True)
                no_pop(outdata, chunksize, False)
                self.current_frame = frames - leftover
            else:
                self.current_frame += frames

            # Last frame fit end of audio perfectly, we need to unpop left next
            if self.current_frame == 0:
                self.trans_left = True

            # Transformations:
            outdata[:] = self.transformation(self.gain * outdata)

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
            if c == "q":
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
