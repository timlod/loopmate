import ctypes
import multiprocessing as mp
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal as sig

from loopmate import config
from loopmate.circular_array import query_circular
from loopmate.loop import Audio
from loopmate.utils import (
    EMA_MinMaxTracker,
    PeakTracker,
    SharedInt,
    StreamTime,
    magsquared,
    samples_to_frames,
    tempo_frequencies,
)


class Recording:
    """
    Class that encapsulates several actions which can be performed on recorded
    audio, stored in a circular array.

    Allow multiple recordings at the same time by means of dicts?
    """

    def __init__(self, rec, loop_length=None):
        self.rec = rec
        # Flag to signal whether we should be recording
        self.started = False
        self.loop_length = loop_length

    def start(self, callback_time, t):
        self.rec_start, frames_since = self.recording_event(callback_time, t)

    def stft(self):
        # Method to run inside AP1
        self.rec.fft()

    def recording_event(
        self, callback_time: StreamTime, t: float
    ) -> (int, int):
        """Return frame in rec that aligns with time t, as well as the number
        of frames that passed since callback_time.

        :param callback_time: StreamTime of the current callback
        :param t: sd.Stream.time to compute the frame for
        """
        frames_since = round(callback_time.timediff(t) * config.sr)
        return (
            self.rec.counter
            + frames_since
            + round(callback_time.input_delay * config.sr)
        ), frames_since


def channels_to_int(channels: tuple) -> int:
    """Map a recording channel mapping to an integer.  Supports at most 8
    concurrent channels, where the maximum channel index is 254.

    Uses a 64 bit integer to encode channel numbers, using every byte to encode
    one channel.  If we don't add 1, we will miss the zeroth channel for
    configurations like (0, 1), (0, 4), etc., which will be very common.

    :param channels: sequence of channel numbers to map
    """
    assert len(channels) <= 8, "There can be at most 8 channels"

    result = 0
    for channel in channels:
        assert channel < 255, "Channel numbers must be at most 254!"
        result <<= 8  # Make room for the next channel number (8 bits)
        result |= channel + 1
    return result


def int_to_channels(value: int) -> list[int]:
    """Maps the output of channels_to_int back into a list of channel numbers.

    Starts reading the last 8 bits of the int64 and shifts one byte to the
    right as long as the result is not 0.

    :param value: integer output of channels_to_int
    """
    channels = []
    while value > 0:
        channel = value & 0xFF  # Extract the least significant 8 bits
        channels.insert(0, channel - 1)
        value >>= 8  # Shift the value to the right by 8 bits
    return channels


def make_recording_struct(
    N, channels, analysis=False, int_type=ctypes.c_int64
):
    n_fft = 2048
    hop_length = 256
    N_stft = int(np.ceil(N / hop_length))
    tg_win_len = 384
    N_onset_env = int(np.ceil(N / hop_length))
    analysis_add = [
        ("stft_counter", int_type),
        ("stft", ctypes.c_float * 2 * ((1 + int(n_fft / 2) * N_stft))),
        ("onset_env", ctypes.c_float * N_onset_env),
        ("mov_max", ctypes.c_float * N_onset_env),
        ("mov_avg", ctypes.c_float * N_onset_env),
        ("tg", ctypes.c_float * tg_win_len * N_onset_env),
    ]

    class CRecording(ctypes.Structure):
        _fields_ = [
            ### Everything before write_counter is used for IPC
            # Flag to trigger stft
            ("do_stft", ctypes.c_bool),
            ## Indicate to analysis thread that an action should be performed
            # Number of recording (to allow for multiple simultaneous
            # recordings)
            ("recording_number", int_type),
            # Channels to record (see channels_to_int)
            ("record_channels", int_type),
            # Start index counter
            ("recording_start", int_type),
            # End index counter
            ("recording_end", int_type),
            # Flag to indicate that analysis thread has completed
            # Potentially just use the integer and have 0 signal not ready
            ("result_ready", ctypes.c_bool),
            # Type of result that can be taken
            ("result_type", int_type),
            # Result? array?
            ("write_counter", int_type),
            ("counter", int_type),
            ("data", ctypes.c_float * (N * channels)),
        ] + (analysis_add if analysis else [])

    return CRecording


class RecMain:
    def __init__(self, N, channels, name="recording"):
        cstruct = make_recording_struct(N, channels)
        self.shm = SharedMemory(
            name=name, create=True, size=ctypes.sizeof(cstruct)
        )
        self.data = cstruct.from_buffer(self.shm.buf)
        self.ca = CircularArray(
            np.ndarray(
                (N, channels),
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.data.offset],
            )
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shm.unlink()


class RecAnalysis:
    def __init__(self, N, channels, name="recording", poll_time=0.0001):
        cstruct = make_recording_struct(N, channels)
        self.shm = SharedMemory(
            name=name, create=False, size=ctypes.sizeof(cstruct)
        )
        self.data = cstruct.from_buffer(self.shm.buf)
        self.ca = CircularArraySTFT(
            np.ndarray(
                (N, channels),
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.data.offset],
            )
        )

    def run(self):
        while True:
            while not self.do_stft:
                time.sleep(self.poll_time)
            self.ca.fft()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shm.close()


class RecA:
    def bpm_quantize_start(self):
        pass

    def get_last_impulse(self):
        pass


class CircularArray:
    """
    Simple implementation of an array which can be indexed and written to in a
    wrap-around fashion.
    """

    def __init__(self, data: np.ndarray, write_counter=0, counter=0, axis=0):
        """
        Initialize CircularArray given numpy array and counters.  Can be backed
        by shared memory.

        :param data: numpy array
        :param write_counter: can be wrapped in a SharedInt
        :param counter: can be wrapped in a SharedInt
        :param axis: axis along which to wrap the array.  Needs to be first or
            last axis (0 or -1)!
        """
        self.data = data
        assert axis in (0, -1), "Axis needs to be either 0 or -1!"
        self.axis = axis
        self.N = data.shape[axis]
        self.write_counter = write_counter
        self.counter = counter

    def query(self, i: slice | int, out=None):
        """Return n samples.  Note: returns a copy of the requested data
        (unless we specify the output array, in which case it writes a copy
        into it)!

        :param i: index or slice of samples to return.  Needs to satisfy
                  -self.N < start < stop <= 0.  Ignores slice step.
        :param out: array to place the samples into.  Can be used to re-use an
            array of loop_length for sample storage to avoid extra memory
            copies.
        """
        if isinstance(i, int):
            if self.axis == 0:
                return self.data[self.index_offset(i)]
            else:
                return self.data[..., self.index_offset(i)]
        return query_circular(
            self.data, i, int(self.write_counter), out, axis=self.axis
        )

    def __getitem__(self, i):
        """Get samples from this array. This returns a copy.

        :param i: slice satisfying -self.N < start < stop <= 0. Can't use step.
        """
        return self.query(i)

    def index_offset(self, offset: Union[int, np.ndarray]):
        i = self.write_counter + offset
        if isinstance(offset, np.ndarray):
            return np.where(
                i > self.N,
                i % self.N,
                np.where(i < 0, self.N + i, i),
            )
        else:
            if i > self.N:
                return i % self.N
            elif i < 0:
                return self.N + i
            else:
                return i

    def frames_since(self, c0):
        return self.counter - c0

    def write(self, arr, increment=True):
        """Write to this circular array.

        :param arr: array to write
        :param increment: whether to increment counters.  Use False only if
            counters are shared and should only be incremented once each
            timestep!
        """
        n = arr.shape[self.axis]
        arr_i = 0

        l_i = 0 + self.write_counter
        if increment:
            self.counter += n
            self.write_counter += n
        if self.write_counter >= self.N:
            arr_i = self.N - l_i
            if self.axis == 0:
                self.data[l_i:] = arr[:arr_i]
            elif self.axis == -1:
                self.data[..., l_i:] = arr[..., :arr_i]
            self.write_counter %= self.N
            l_i = 0
        if self.axis == 0:
            self.data[l_i : self.write_counter] = arr[arr_i:]
        else:
            self.data[..., l_i : self.write_counter] = arr[..., arr_i:]

    def __repr__(self):
        return (
            self.data.__repr__()
            + f"\nN: {self.N}, i: {self.write_counter}, c: {self.counter}"
        )


class CircularArraySTFT(CircularArray):
    def __init__(self, N, channels=2, n_fft=2048, hop_length=256, sr=44100):
        super().__init__(N, channels, create=False)
        self.N_stft = int(np.ceil(N / hop_length))
        self.stft = np.zeros(
            (1 + int(n_fft / 2), self.N_stft),
            dtype=np.complex64,
        )

        self.stft_counter = 0
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = sig.windows.hann(n_fft).astype(np.float32)
        self.sr = sr

        # Onset tracking vars
        self.onset_env = np.zeros(
            int(np.ceil(N / hop_length)), dtype=np.float32
        )
        self.tg_win_len = 384
        self.tg_pad = 2 * self.tg_win_len - 1
        self.tg_window = sig.windows.hann(self.tg_win_len).astype(np.float32)
        self.tg = np.zeros(
            (self.tg_win_len, len(self.onset_env)),
            dtype=np.float32,
        )
        self.onset_env_minmax = EMA_MinMaxTracker(
            min0=0, minmin=0, max0=1, alpha=0.001
        )
        self.logspec_minmax = EMA_MinMaxTracker(
            max0=10, minmax=0, alpha=0.0005
        )

        self.mov_max = np.zeros(int(np.ceil(N / hop_length)), dtype=np.float32)
        self.mov_avg = np.zeros(int(np.ceil(N / hop_length)), dtype=np.float32)

        # Onset detection parameters
        pre_max = int(0.03 * sr // hop_length)
        post_max = int(0.0 * sr // hop_length + 1)
        self.max_length = pre_max + post_max
        max_origin = int(np.ceil(0.5 * (pre_max - post_max)))
        self.max_offset = (self.max_length // 2) - max_origin

        pre_avg = int(0.1 * sr // hop_length)
        post_avg = int(0.1 * sr // hop_length + 1)
        self.avg_length = pre_avg + post_avg
        avg_origin = int(np.ceil(0.5 * (pre_avg - post_avg)))
        self.avg_offset = (self.avg_length // 2) - avg_origin

        self.wait = int(0.03 * sr // hop_length)
        self.delta = 0.07

        offset = (
            self.avg_offset
            if self.avg_offset > self.max_offset
            else self.max_offset
        )
        self.peaks = PeakTracker(self.N_stft, offset)

        self.tf = tempo_frequencies(self.tg_win_len, hop_length, sr=sr)
        self.bpm_logprior = (
            -0.5 * ((np.log2(self.tf) - np.log2(100)) / 1.0) ** 2
        )[:, None]

    def index_offset_stft(self, offset: Union[int, np.ndarray]):
        if isinstance(offset, np.ndarray):
            i = self.stft_counter + offset
            return np.where(
                i > self.N_stft,
                i % self.N_stft,
                np.where(i < 0, self.N_stft + i, i),
            )
        else:
            if (i := self.stft_counter + offset) >= self.N_stft:
                return i % self.N_stft
            elif i < 0:
                return self.N_stft + i
            else:
                return i

    def fft(self):
        # Make sure this runs at every update!

        # TODO: If this starts lagging, make sure we don't just ignore
        # segments, but transform everything since this was last called
        # To do that, we'd have to vectorize this, which adds complexity :/
        # Better to just optimize it such that it runs quickly
        self.stft[:, self.stft_counter] = np.fft.rfft(
            self.window * self[-self.n_fft :].mean(-1)
        )
        self.detect_onsets()
        self.tempogram()
        self.stft_counter += 1
        if self.stft_counter >= self.stft.shape[1]:
            self.stft_counter = 0
        # print(f"{self.stft_counter=}")

    def tempogram(self):
        oe_slice = query_circular(
            self.onset_env, slice(-self.tg_win_len, None), self.stft_counter
        )
        tg = np.fft.irfft(
            magsquared(np.fft.rfft(self.tg_window * oe_slice, n=self.tg_pad)),
            n=self.tg_pad,
        )[: self.tg_win_len]
        self.tg[:, self.stft_counter] = tg / (tg.max() + 1e-10)

    def tempo(self, tg, agg=np.mean):
        # From librosa.feature.rhythm
        if agg is not None:
            tg = agg(tg, axis=-1, keepdims=True)
        best_period = np.argmax(
            np.log1p(1e6 * tg) + self.bpm_logprior, axis=-2
        )
        return np.take(self.tf, best_period)

    def detect_onsets(self):
        # Potentially move over to fft
        mag = magsquared(self.stft[:, self.stft_counter])
        magm1 = magsquared(self.stft[:, self.index_offset_stft(-1)])
        # Convert to DB
        s = 10.0 * np.log10(np.maximum(1e-10, mag))
        self.logspec_minmax.add_sample(s.max())
        s = np.maximum(s, self.logspec_minmax.max_val - 80)
        sm1 = 10.0 * np.log10(np.maximum(1e-10, magm1))
        sm1 = np.maximum(sm1, self.logspec_minmax.max_val - 80)
        # Aggregate frequencies
        onset_env = np.maximum(0.0, s - sm1).mean()

        # Normalize and add to self
        self.onset_env_minmax.add_sample(onset_env)
        self.onset_env[
            self.stft_counter
        ] = self.onset_env_minmax.normalize_sample(onset_env)

        mov_max_cur = self.index_offset_stft(-self.max_offset)
        self.mov_max[mov_max_cur] = np.max(
            query_circular(
                self.onset_env,
                slice(-self.max_length, None),
                self.stft_counter,
            )
        )
        mov_avg_cur = self.index_offset_stft(-self.avg_offset)
        self.mov_avg[mov_avg_cur] = np.mean(
            query_circular(
                self.onset_env,
                slice(-self.avg_length, None),
                self.stft_counter,
            )
        )
        # The average filter is usually wider than the max filter, and
        # determines our lag wrt. onset detection.
        cur = mov_avg_cur if self.avg_offset > self.max_offset else mov_max_cur
        detect = self.onset_env[cur] * (
            self.onset_env[cur] == self.mov_max[cur]
        )
        detect *= detect >= self.mov_avg[cur] + self.delta
        if detect:
            if -self.avg_offset > self.peaks.last + self.wait:
                # Found an onset
                self.peaks.add_element()

        self.peaks.step()

    def bpm_quantize(self, start, end):
        # start and end are negative
        start_f = samples_to_frames(start)
        end_f = samples_to_frames(end)
        tg = query_circular(
            self.tg, slice(start_f, end_f), self.stft_counter, axis=-1
        )
        onsets = np.array(self.peaks.relative)
        # onsets = onsets[(onsets >= start_f) & (onsets <= end_f)]
        # TODO: Perhaps allow for tempo to change slightly? maybe not
        bpm = self.tempo(tg)
        print(bpm)
        beat_len = (self.sr // self.hop_length) // (bpm / 60)
        start_diff = np.abs(onsets - start_f)
        potential_start = (
            onsets[(onsets > start_f) & (onsets < start_f + 4 * beat_len)]
            - beat_len
        )
        ps_diff = np.abs(potential_start - start)
        lenience = round(self.sr // self.hop_length * 0.1)
        if start_diff[(i := start_diff.argmin())] < lenience:
            print(f"{start_f=} -> {onsets[i]=}")
            start_f = onsets[i]
            start_move = start_diff[i]
        elif ps_diff[(i := ps_diff.argmin())] < lenience:
            print(f"{start_f=} -> {potential_start[i]=}")
            start_f = potential_start[i]
            start_move = ps_diff[i]
        else:
            start_move = 0

        # print(f"{onsets=}\n{end_f=}\n{beat_len=}")

        end_diff = np.abs(onsets - end_f)
        potential_end = (
            onsets[(onsets < end_f) & (onsets > end_f - 4 * beat_len)]
            + beat_len
        )
        ps_diff = np.abs(potential_end - end_f)
        print(f"{end_diff=}\n{potential_end=}\n{ps_diff=}")

        if end_diff[(i := end_diff.argmin())] < lenience:
            print(f"{end_f=} -> {onsets[i]=}")
            end_f = onsets[i]
            end_move = end_diff[i]
        elif ps_diff[(i := ps_diff.argmin())] < lenience:
            print(f"{end_f=} -> {potential_end[i]=}")
            end_f = potential_end[i]
            end_move = ps_diff[i]
        else:
            end_move = 0
        return start_f, start_move, end_f, end_move

    def write(self, arr):
        super().write(arr)
        self.fft()
