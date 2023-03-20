import ctypes
from collections import deque
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Union

import numpy as np
import scipy as sp
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


def query_circular(
    data: np.ndarray,
    idx_slice: slice,
    counter: int,
    out: Optional[np.ndarray] = None,
):
    """Return n samples, backwards from current counter.  Note: returns a copy
    of the requested data

    :param data: array to make a circular query into
    :param idx_slice: slice of samples to return.  Needs to satisfy -self.N <
        start < stop <= 0.  Ignores slice step.
    :param counter: index pointing to the latest entry in data (counter + 1
        will be the last entry)
    :param out: array to place the samples into.  Can be used to re-use an
        array of loop_length for sample storage to avoid extra memory copies.
    """
    assert isinstance(
        idx_slice, slice
    ), f"Use slice for indexing! (Got {idx_slice})"
    start, stop = idx_slice.start or 0, idx_slice.stop or 0
    N = len(data)
    assert (
        -N < start < stop <= 0
    ), f"Can only slice at most N ({N}) items backward on!"
    l_i = counter + start
    r_i = counter + stop
    if l_i < 0 <= r_i:
        return np.concatenate((data[l_i:], data[:r_i]), out=out)
    else:
        if out is not None:
            out[:] = data[l_i:r_i]
            return out
        else:
            return data[l_i:r_i].copy()


class CircularArray:
    """
    Simple implementation of an array which can be indexed and written to in a
    wrap-around fashion.
    """

    def __init__(self, N, channels=2):
        self.data = np.zeros((N, channels), dtype=np.float32)
        self.N = N
        self.write_counter = 0
        # Use to compute differences in samples between two points in time
        self.counter = 0

    def query(self, i: slice, out=None):
        """Return n samples.  Note: returns a copy of the requested data
        (unless we specify the output array, in which case it writes a copy
        into it)!

        :param i: slice of samples to return.  Needs to satisfy -self.N < start
                  < stop <= 0.  Ignores slice step.
        :param out: array to place the samples into.  Can be used to re-use an
            array of loop_length for sample storage to avoid extra memory
            copies.
        """
        return query_circular(self.data, i, self.write_counter, out)

    def __getitem__(self, i):
        """Get samples from this array. This returns a copy.

        :param i: slice satisfying -self.N < start < stop <= 0. Can't use step.
        """
        return self.query(i)

    def index_offset(self, offset):
        if (i := self.write_counter + offset) > self.N:
            return i % self.N
        elif i < 0:
            return self.N + i
        else:
            return i

    def frames_since(self, c0):
        return self.counter - c0

    def write(self, arr):
        """Write to this circular array.

        :param arr: array to write
        """
        n = len(arr)
        arr_i = 0

        # left index - use 0 + to avoid copying the reference to SharedInt
        l_i = 0 + self.write_counter
        self.write_counter += n
        # Wrap around if we cross the boundary in data
        if self.write_counter >= self.N:
            arr_i = self.N - l_i
            self.data[l_i:] = arr[:arr_i]
            self.write_counter %= self.N
            l_i = 0
        print(f"{l_i=}, {self.write_counter=}, {arr_i=}")
        self.data[l_i : self.write_counter] = arr[arr_i:]
        self.counter += n

    def make_shared(self, name="recording", create=False):
        self.shm = SharedMemory(
            name=name, create=create, size=self.data.nbytes + 16
        )
        data = np.ndarray(
            self.data.shape, dtype=self.data.dtype, buffer=self.shm.buf[16:]
        )
        write_counter = SharedInt(self.shm, 0)
        counter = SharedInt(self.shm, 8)
        if create:
            data[:] = self.data[:]
            write_counter.value = self.write_counter
            counter.value = self.counter
        self.data = data
        self.write_counter = write_counter
        self.counter = counter

    def stop_sharing(self, unlink=True):
        assert isinstance(self.write_counter, SharedInt), "Not sharing!"
        self.data = self.data.copy()
        self.write_counter = self.write_counter.value
        self.counter = self.counter.value
        self.shm.close()
        if unlink:
            # Ignore if already closed by another process
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
        self.shm = None

    def __repr__(self):
        return self.data.__repr__() + f"\ni: {self.write_counter}"


class EMA_MinMaxTracker:
    def __init__(self, alpha=0.0001, eps=1e-10):
        self.alpha = alpha
        self.eps = eps
        self.min_val = 0
        self.max_val = float("-inf")

    def add_sample(self, sample):
        # Update min_val and max_val using exponential moving average
        # sample = abs(sample)
        if sample < self.min_val:
            # print(f"new min: {sample}")
            self.min_val = sample
        else:
            self.min_val = (
                self.min_val * (1 - self.alpha) + sample * self.alpha
            )

        if sample > self.max_val:
            # print(f"new max: {sample}")
            self.max_val = sample
        else:
            self.max_val = (
                self.max_val * (1 - self.alpha) + sample * self.alpha
            )

    def normalize_sample(self, sample):
        # print(f"{self.min_val=}, {self.max_val=}, ")
        if self.max_val == self.min_val:
            return 0  # Avoid division by zero
        sample -= self.min_val
        return sample / (self.max_val + self.eps)

class PeakTracker:
    def __init__(self, N, offset=0):
        self.N = N
        self.absolute = deque()
        self.relative = deque()
        self.current_step = 0
        self.offset = offset

    def add_element(self):
        self.absolute.append(self.current_step)
        self.relative.append(self.offset)

    def decrement(self):
        self.current_step -= 1
        while self.absolute and self.absolute[0] - self.N > self.current_step:
            self.absolute.popleft()
            self.relative.popleft()
        for i in range(len(self.relative)):
            self.relative[i] -= 1
class CircularArraySTFT(CircularArray):
    def __init__(self, N, channels=2, n_fft=2048, hop_length=256):
        super().__init__(N, channels)
        self.stft = np.zeros(
            (1 + int(n_fft / 2), int(np.ceil(N / hop_length))),
            dtype=np.complex64,
        )
        self.stft_counter = 0
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = sig.windows.hann(n_fft)

    def fft(self):
        # Make sure this runs at every update!
        bit = self[-self.n_fft :].mean(-1)
        self.stft[:, self.stft_counter] = np.fft.rfft(self.window * bit)
        self.stft_counter += 1
        if self.stft_counter > self.stft.shape[1]:
            self.stft_counter = 0

    def write(self, arr):
        super().write(arr)
        self.fft()


class SharedInt:
    def __init__(self, shared_memory_object, offset=0, value=None):
        self._int = ctypes.c_int64.from_buffer(
            shared_memory_object.buf, offset
        )
        if value is not None:
            assert isinstance(value, int)
            self.value = value

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __iadd__(self, other):
        self.value += other
        return self

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __isub__(self, other):
        self.value -= other
        return self

    def __floordiv__(self, other):
        return self.value // other

    def __ifloordiv__(self, other):
        self.value //= other
        return self

    def __rmod__(self, other):
        return other % self.value

    def __mod__(self, other):
        return self.value % other

    def __imod__(self, other):
        self.value %= other
        return self

    def __eq__(self, other):
        return self.value == other

    def __le__(self, other):
        return self.value <= other

    def __ge__(self, other):
        return self.value >= other

    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __index__(self):
        return self.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"SharedInt({self.value})"

    @property
    def value(self):
        return self._int.value

    @value.setter
    def value(self, new_value):
        self._int.value = new_value
