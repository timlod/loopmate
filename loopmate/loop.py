from __future__ import annotations

import asyncio
import queue
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import KW_ONLY, dataclass, field
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

blend_windowsize = int(config.blend_length * config.sr)
RAMP = np.linspace(1, 0, blend_windowsize, dtype=np.float32)[:, None]
POP_WINDOW = sig.windows.hann(int(config.sr * config.blend_length))[:, None]


@dataclass
class Audio:
    # store info about different audio files to loop, perhaps with actions
    # around them as well, i.e. meter, bpm, etc.
    audio: np.ndarray
    loop_length: int | None = None
    pos_start: int = 0
    current_frame: int = 0

    def __post_init__(self):
        # Remove pop at loop edge
        self.n, self.channels = self.audio.shape
        if self.loop_length is None:
            self.loop_length = self.n

        n_pw = len(POP_WINDOW) // 2
        self.audio[:n_pw] *= POP_WINDOW[:n_pw]
        self.audio[-n_pw:] *= POP_WINDOW[-n_pw:]
        self.n_frames = np.ceil(self.n / config.blocksize)
        self.current_frame = 0

        left = self.pos_start
        self.n += left
        if self.n <= self.loop_length:
            right = self.loop_length - self.n
            self.n_loop_iter = 1
        else:
            # Need to zero-pad right to a power of 2 amount of loop iterations
            self.n_loop_iter = int(
                2 * np.ceil(np.log2(self.n / self.loop_length))
            )
            right = self.n_loop_iter * self.loop_length - self.n

        self.audio = np.concatenate(
            [
                np.zeros((left, self.channels), dtype=np.float32),
                self.audio,
                np.zeros((right, self.channels), dtype=np.float32),
            ]
        )
        self.n += right
        self.actions = Actions(self)

    def get_n(self, frames: int):
        """Return the next batch of audio in the loop

        :param frames: number of audio samples to return
        """
        leftover = self.n - self.current_frame
        chunksize = min(leftover, frames)

        if leftover <= frames:
            out = self.audio[
                np.r_[
                    self.current_frame : self.current_frame + chunksize,
                    : frames - leftover,
                ]
            ]
            self.current_frame = frames - leftover
        else:
            out = self.audio[
                self.current_frame : self.current_frame + chunksize
            ]
            self.current_frame += frames

        self.actions.run(out, self.current_frame)
        return out


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
        self.current_sample = self.n
        self.loop = False

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

    def cancel(self):
        """Stops effect over the next buffer(s).  Fades out to avoid audio
        popping, and may thus take several callbacks to complete.
        """
        self.blend = Blender(left_right=False)
        self.current_sample = self.n - self.blend.n
        self.loop = False


class Start(Action):
    def __init__(self, start: int, priority: int = 100):
        """Initialize effect which will fade in at a certain frame.

        :param start: start effect at this frame (inside looped audio)
        :param n: length of looped audio
        :param transformation: callable of form f(outdata) which returns an
            ndarray of the same size as outdata
        :param priority: indicate priority at which to queue this action
        """
        blend = Blender(left_right=False)
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
        :param n: length of looped audio
        :param transformation: callable of form f(outdata) which returns an
            ndarray of the same size as outdata
        :param priority: indicate priority at which to queue this action
        """
        blend = Blender()
        super().__init__(start, start + blend.n, loop=False)
        self.blend = blend

    def do(self, data):
        data_trans = data * 0.0
        data_trans = self.blend(data, data_trans)
        data[:] = data_trans


@dataclass
class Actions:
    # keeps and maintains a queue of actions that are fired in the callback
    loop: Loop
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


class Loop:
    def __init__(self, anchor: Audio | None = None):
        self.audios = []
        self.anchor = anchor
        if anchor is not None:
            self.audios.append(anchor)

        # Always record the latest second buffers so we can look a little into
        # the past when recording
        self.recent_audio = queue.deque(
            maxlen=int(np.ceil(config.sr / config.blocksize))
        )
        self.recording = []

        # Global actions applied to fully mixed audio
        self.actions = Actions(self)

        self.stream = sd.Stream(
            samplerate=config.sr,
            device=config.device,
            channels=2,
            callback=self._get_callback(),
            latency=config.latency,
            blocksize=config.blocksize,
        )
        self.rtimes = []
        self.rf = False
        self.last_out = None
        self.frame_times = None

    def add_track(self, audio):
        if len(self.audios) == 0:
            self.anchor = Audio(audio)
            self.audios.append(self.anchor)
        else:
            self.audios.append(Audio(audio, self.anchor.loop_length))

    def _get_callback(self):
        """
        Creates callback function for this loop.
        """

        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            current_frame = self.anchor.current_frame
            self.frame_times = (
                current_frame,
                time.currentTime,
                time.inputBufferAdcTime,
                time.outputBufferDacTime,
            )

            if self.rf:
                self.recording.append(indata.copy())
                self.rtimes.append(time.inputBufferAdcTime)
            self.recent_audio.append((time.inputBufferAdcTime, indata.copy()))

            outdata[:] = 0.0
            for audio in self.audios:
                a = audio.get_n(frames)
                outdata[:] += a

            if self.anchor is not None:
                self.actions.run(outdata, current_frame)

            # Store last output buffer to potentially send a slightly delayed
            # version to headphones (to match the speaker sound latency)
            self.last_out = outdata.copy()

        return callback

    async def start(self):
        self.stream.stop()
        # self.current_frame = 0
        if self.anchor is not None:
            self.actions.actions.append(Start(self.anchor.current_frame))
        self.stream.start()
        sd.sleep(200)

    def stop(self):
        if self.anchor is not None:
            print(f"Stopping stream action. {self.anchor.current_frame}")
            self.actions.actions.append(Stop(self.anchor.current_frame))

    def time_frame(self, t):
        """Compute the approximate frame that was played at a given time.

        Specifically, this computes the difference between the given time and
        the time the current frame will be played by the DAC, and converts to
        number of frames by using the sample rate.

        :param current_frame: frame that we will compute relative to, should
            probably be self.anchor.current_frame
        :param time: time we want to check, should probably be self.stream.time

        :returns:
        """
        play_delay = self.frame_times[3] - t
        frames_back = int(play_delay * config.sr)
        frame = self.frame_times[0] - frames_back
        if frame < 0:
            # Time was at the end of the loop
            return self.anchor.loop_length + frame
        else:
            return frame
    def record(self):
        # TODO:
        rt = self.stream.time
        print(f"Stream time: {self.stream.time}")
        self.rf = not self.rf
        if self.rf:
            print("Starting recording")
            while True:
                try:
                    t, a = self.recent_audio.popleft()
                except IndexError:
                    break
                if rt - 0.005 < t:
                    self.recording.append(a)
                    self.rtimes.append(t)
        if not self.rf:
            print("Stopping recording")
            while True:
                try:
                    t = self.rtimes.pop()
                except IndexError:
                    break
                if rt + 0.005 < t:
                    self.recording.pop()
            self.add_track(np.concatenate(self.recording))
            self.rtimes.clear()
            self.recording.clear()
