# Modify/mix audio in separate process, once that's done, replace the audio in
# the callback thread, such that the audio thread always just indexes the array
# For mute, probably can only run action on mixed thread, instead of just the
# one which is being muted

# Store didchange variable and audio_prev - if didchange, add fade
from __future__ import annotations

import asyncio
import queue
import sys
import termios
import tty
from collections import deque
from dataclasses import KW_ONLY, dataclass, field

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
from scipy import signal as sig

from loopmate import config
from loopmate.loop import Action, Effect, Start, Stop

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

        # Always record the latest 2*latency amount of buffers so we can look a
        # little into the past when recording
        self.recent_audio = queue.deque(
            maxlen=int(
                np.ceil(config.latency * config.sr * 2 / config.blocksize)
            )
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

            if self.rf:
                self.recording.append(indata.copy())
                self.rtimes.append(time.inputBufferAdcTime)
            self.recent_audio.append((time.inputBufferAdcTime, indata.copy()))

            outdata[:] = 0.0
            for audio in self.audios:
                a = audio.get_n(frames)
                outdata[:] += a

            if self.anchor is not None:
                self.actions.run(outdata, self.anchor.current_frame)

            # print(
            #     time.currentTime, time.currentTime - time.outputBufferDacTime
            # )

        return callback

    async def start(self):
        self.stream.stop()
        # self.current_frame = 0
        if self.anchor is not None:
            self.actions.actions.append(Start(self.anchor.current_frame))
        self.stream.start()
        sd.sleep(200)

    def stop(self):
        print("Stopping stream action.")
        if self.anchor is not None:
            self.actions.actions.append(Stop(self.anchor.current_frame))

    def record(self):
        rt = self.stream.time
        print(f"Stream time: {self.stream.time}")
        self.rf = not self.rf
        # TODO: CHECK LOGIC
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

    def save_times(self):
        pd.DataFrame({"times": self.times, "ctimes": self.ctimes}).to_csv(
            "times.csv"
        )


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
    # print(sd.query_devices())
    piano, _ = sf.read("../data/piano.wav")
    tom, _ = sf.read("../data/tom.wav")
    tom = tom[:, None]
    piano = piano[: len(tom) * 2]
    # sweep = 0.2 * chirp(200, 1000, 2, method="linear", sr=sr)
    # loop = Loop(sweep[:, None])
    # loop = Loop(Audio(piano))
    # loop.add_track(tom)
    # loop = Loop(Audio(tom))
    loop = Loop()
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
                if loop.actions.mute is None:
                    print("MUTING")
                    loop.actions.mute = Effect(
                        loop.anchor.current_frame,
                        loop.anchor.n,
                        lambda x: x * 0.0,
                    )
                else:
                    loop.actions.mute.cancel()
            if c == "o":
                loop.stop()
            if c == "r":
                loop.record()
            if c == "a":
                loop.gain = 0.5
            if c == "b":
                loop.gain = 1.0
            if c == "c":
                # schedule stop task at end of loop
                loop.save_times()
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
