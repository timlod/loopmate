from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import signal as sig


def resample(x: np.ndarray, sr_source: int, sr_target: int):
    """Resample signal to target samplerate

    :param x: audio file to resample of shape (n_samples, channels)
    :param sr_source: samplerate of x
    :param sr_target: target samplerate
    """
    c = sr_target / sr_source
    n_target = int(np.round(len(x) * c))
    return sig.resample(x, n_target)


@dataclass
class Metre:
    bpm: float
    beats: int
    divisions: int

    @property
    def bps(self):
        return self.bpm / 60

    def get_metronome(self, sr: int):
        clave, msr = sf.read("../data/clave.wav", dtype=np.float32)
        if sr != msr:
            clave = resample(clave, msr, sr)
        beat = int(sr / self.bps)
        out = np.zeros(int(beat * self.beats), dtype=np.float32)
        for i in range(self.beats):
            out[i * beat : i * beat + len(clave)] = clave
        return out


class StreamTime:
    def __init__(self, time, frame, n_frames):
        self.frame = frame
        self.n_frames = n_frames
        if isinstance(time, list):
            self.current = time[0]
            self.input = time[1]
            self.output = time[2]
        else:
            self.current = time.currentTime
            self.input = time.inputBufferAdcTime
            self.output = time.outputBufferDacTime

    @property
    def full_delay(self):
        return self.output - self.input

    def full_delay_frames(self, sr):
        return int(np.round(self.full_delay * sr))

    @property
    def input_delay(self):
        return self.current - self.input

    @property
    def output_delay(self):
        return self.current - self.output

    def timediff(self, t):
        return t - self.current

    def __repr__(self):
        return f"StreamTime({self.current}, {self.input}, {self.output}, {self.frame})"


class CircularArray:
    """
    Simple implementation of an array which can be indexed and written to in a
    wrap-around fashion.
    """

    def __init__(self, N, channels=2):
        self.data = np.zeros((N, channels), dtype=np.float32)
        self.N = N
        self.i = 0

    def __getitem__(self, n: int, out=None):
        """Return the last n samples.  Note: returns a view if we don't need to
        wrap around, and a copy otherwise (unless we specify the output array)!

        :param n: number of samples to return.  Needs to be < N
        :param out: array to place the last n samples into.  Can be used to
            re-use an array of loop_length for sample storage to avoid extra
            memory copies.
        # Use to compute differences in samples between two points in time
        self.counter = 0
        """
        assert n <= self.N, f"Can't query more than N({self.N}) samples!"
        l_i = self.i - n
        if l_i < 0:
            return np.concatenate(
                (self.data[l_i:], self.data[: self.i]), out=out
            )
        else:
            if out is not None:
                out[:] = self.data[l_i : self.i]
                return out
            else:
                return self.data[l_i : self.i]

    def write(self, arr):
        """Write to this circular array.

        :param arr: array to write
        """
        n = len(arr)
        arr_i = 0

        # right index
        r_i = self.i + n
        # Wrap around if we cross the boundary in data
        if r_i > self.N:
            arr_i = self.N - self.i
            self.data[self.i :] = arr[:arr_i]
            r_i = r_i % self.N
            self.i = 0
        self.data[self.i : r_i] = arr[arr_i:]
        self.i = r_i
        self.counter += n

    def __repr__(self):
        return self.data.__repr__() + f"\ni: {self.i}"
