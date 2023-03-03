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
    remove_pop: bool = True

    def __post_init__(self):
        # Remove pop at loop edge
        self.n, self.channels = self.audio.shape
        if self.loop_length is None:
            self.loop_length = self.n

        self.n_frames = np.ceil(self.n / config.blocksize)

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

        if self.remove_pop:
            n_pw = len(POP_WINDOW) // 2
            self.audio[:n_pw] *= POP_WINDOW[:n_pw]
            self.audio[-n_pw:] *= POP_WINDOW[-n_pw:]

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
        :param n: length of looped audio
        :param transformation: callable of form f(outdata) which returns an
            ndarray of the same size as outdata
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
        :param n: length of looped audio
        :param transformation: callable of form f(outdata) which returns an
            ndarray of the same size as outdata
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


class Loop:
    def __init__(self, anchor: Audio | None = None):
        self.audios = []
        self.new_audios = queue.Queue()
        self.anchor = anchor
        if anchor is not None:
            self.audios.append(anchor)

        # Always record the latest second buffers so we can look a little into
        # the past when recording
        self.recent_audio = queue.deque(
            maxlen=int(np.ceil(config.sr / config.blocksize))
        )
        self.recording = None

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
        self.last_out = None
        self.frame_times = None

    def add_track(self, audio):
        if len(self.audios) == 0:
            if not isinstance(audio, Audio):
                audio = Audio(audio)
            self.anchor = audio
            self.audios.append(self.anchor)
        else:
            if not isinstance(audio, Audio):
                audio = Audio(audio, self.anchor.loop_length)
            self.new_audios.put(audio)
            self.audios.append(audio)

    def _get_callback(self):
        """
        Creates callback function for this loop.
        """

        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)

            if self.anchor is None:
                current_frame = 0
            else:
                current_frame = self.anchor.current_frame

            # These times/frame refer to the frame that is processed in this
            # callback
            self.frame_times = (
                current_frame,
                time.currentTime,
                time.inputBufferAdcTime,
                time.outputBufferDacTime,
            )

            # Copy necessary as indata arg is passed by reference
            indata = indata.copy()
            self.recent_audio.append(indata)
            if self.recording is not None:
                self.recording.append(indata)

            outdata[:] = 0.0
            for audio in self.audios:
                a = audio.get_n(frames)
                outdata[:] += a

            if self.anchor is not None:
                self.actions.run(outdata, current_frame)

            # Align current frame for newly put audio
            for i in range(self.new_audios.qsize()):
                audio = self.new_audios.get()
                audio.current_frame = self.anchor.current_frame

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

    def frame_delay(self, t):
        play_delay = self.frame_times[3] - t
        return int(play_delay * config.sr)

    def record(self):
        # TODO: put start time correctly
        t = self.stream.time
        print(
            self.frame_times,
            self.frame_times[1] - self.frame_times[2],
            self.frame_times[1] - self.frame_times[3],
        )
        frames_back = self.frame_delay(t)
        if self.recording is None:
            # self.frame_times[0] (current_frame) will respond to the latest
            # item in recent_audio
            print()
            self.recording = Recording(
                list(self.recent_audio),
                self.frame_times[0],
                frames_back,
                self.anchor,
            )
        else:
            self.add_track(self.recording.finish(frames_back))
            self.recording = None
        print(self.stream.cpu_load)


class Recording:
    def __init__(
        self,
        recordings: list[np.ndarray],
        reference_frame: int = 0,
        start_frame_at: int = 0,
        anchor: Audio | None = None,
    ):
        # TODO:
        # 1. get just recorded array from start to end
        # 2. associate reference markers to those start and end
        lengths = [len(x) for x in recordings]
        # n = sum(lengths)
        # This is the actual recording array index of reference_frame
        at = sum(lengths[:-1])

        start_frame = reference_frame - start_frame_at
        # Quantize to anchor if it exists
        if anchor is not None:
            self.loop_length = anchor.loop_length
            if start_frame < 0:
                start_frame += self.loop_length
            self.start_frame, move = self.quantize(start_frame)
            start_frame_at -= move
        else:
            self.start_frame = 0
            self.loop_length = None

        # Separate actual recording
        start_idx = at - start_frame_at
        recording = np.concatenate(recordings)
        self.end_xfade = recording[start_idx - config.blend_frames : start_idx]
        if len(self.end_xfade) < config.blend_frames:
            self.end_xfade = np.zeros(
                (config.blend_frames, recording.shape[1]), np.float32
            )
        self.recordings = [recording[start_idx:]]

    def append(self, indata):
        self.recordings.append(indata)

    def finish(self, end_frame_at):
        n_final = len(self.recordings[-1])
        recording = np.concatenate(self.recordings)

        at = len(recording) - n_final
        reference_frame = self.start_frame + at
        end_frame = reference_frame - end_frame_at
        if self.loop_length is not None:
            self.end_frame, move = self.quantize(end_frame, False)
            print(f"Moving {move} from {end_frame} to {self.end_frame}")
            end_frame_at -= move
        else:
            self.end_frame = end_frame
            self.loop_length = end_frame

        end_idx = at - end_frame_at
        print(len(recording), at, end_frame_at)
        recording = recording[:end_idx]

        if (self.end_frame % self.loop_length) == 0:
            print(len(recording))
            recording[-config.blend_frames :] = (
                RAMP * recording[-config.blend_frames :]
                + (1 - RAMP) * self.end_xfade
            )
            rp = False
        else:
            rp = True

        audio = Audio(
            recording,
            self.loop_length,
            self.start_frame,
            self.end_frame,
            remove_pop=rp,
        )
        print(audio)
        return audio

    def quantize(self, frame, start=True, lenience=0.2) -> (int, int):
        """Quantize start or end recording marker to the loop boundary if
        within some interval from them.  Also returns difference between
        original frame and quantized frame.

        :param frame: start or end recording marker
        :param start: True for start, or False for end
        :param lenience: quantize if within this many seconds from the loop
            boundary

            For example, for sr=48000, the lenience (at 200ms) is 9600 samples.
            If the end marker is at between 38400 and 57600, it will instead be
            set to 48000, the full loop.
        """
        loop_n, frame_rem = np.divmod(frame, self.loop_length)
        lenience = config.sr * lenience
        if start:
            if frame < lenience:
                return 0, frame
            elif frame > (self.loop_length - lenience):
                return 0, self.loop_length - frame
            else:
                return frame, 0
        else:
            if frame_rem < lenience:
                return loop_n * self.loop_length, frame_rem
            elif frame_rem > (self.loop_length - lenience):
                return (
                    loop_n + 1
                ) * self.loop_length, -self.loop_length + frame_rem
            else:
                return frame, 0

