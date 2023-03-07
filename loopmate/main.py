# Modify/mix audio in separate process, once that's done, replace the audio in
# the callback thread, such that the audio thread always just indexes the array
# For mute, probably can only run action on mixed thread, instead of just the
# one which is being muted

# Store didchange variable and audio_prev - if didchange, add fade
from __future__ import annotations

import asyncio
import sys
import termios
import tty

import numpy as np
import pedalboard
import sounddevice as sd
import soundfile as sf
from scipy import signal as sig

from loopmate import config
from loopmate.actions import Effect
from loopmate.loop import Audio, Loop

# TODO: implement fitting effects as actions


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
    piano, _ = sf.read("../data/piano.wav", dtype=np.float32)
    tom, _ = sf.read("../data/tom.wav", dtype=np.float32)
    tom = tom[:, None]
    piano = piano[: len(tom) * 2]
    # sweep = 0.2 * chirp(200, 1000, 2, method="linear", sr=sr)
    # loop = Loop(sweep[:, None])
    loop = Loop(Audio(piano))
    # loop.add_track(tom)
    # loop = Loop(Audio(tom))
    # loop = Loop()
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
            if c == "d":
                loop.audios.pop()
            if c == "o":
                loop.stop()
            if c == "r":
                await loop.record()
            if c == "a":
                with sd.OutputStream() as s:
                    s.write(np.hstack([tom, tom]))
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
