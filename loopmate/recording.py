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
    N,
    channels,
    analysis=False,
    int_type=ctypes.c_int64,
):
    N_stft = int(np.ceil(N / config.hop_length))
    analysis_add = [
        ("stft_counter", int_type),
        (
            "stft",
            ctypes.c_float * (2 * (1 + int(config.n_fft / 2)) * N_stft),
        ),
        ("onset_env", ctypes.c_float * N_stft),
        ("mov_max", ctypes.c_float * N_stft),
        ("mov_avg", ctypes.c_float * N_stft),
        ("tg", ctypes.c_float * config.tg_win_len * N_stft),
        ("analysis_action", int_type),
        ("quit", ctypes.c_bool),
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
        cstruct = make_recording_struct(N, channels, True)
        self.cstruct = cstruct
        self.shm = SharedMemory(
            name=name, create=True, size=ctypes.sizeof(cstruct)
        )
        self.data = cstruct.from_buffer(self.shm.buf)
        self.ca = CircularArray(
            np.ndarray(
                (N, channels),
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.data.offset :],
            ),
            SharedInt(self.shm, cstruct.write_counter.offset),
            SharedInt(self.shm, cstruct.counter.offset),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("exiting")
        # Without this deletion, we'll get a BufferError, even though
        # https://docs.python.org/3/library/multiprocessing.shared_memory.html#multiprocessing.shared_memory.SharedMemory.close
        # states it's unnecessary
        del (self.data, self.ca)
        self.shm.close()
        self.shm.unlink()
        print("exiting unlinked")


class RecAnalysis:
    def __init__(self, N, channels, name="recording", poll_time=0.0001):
        # TODO: take N from config or also kwarg other config items
        self.poll_time = poll_time

        self.N_stft = int(np.ceil(N / config.hop_length))
        cstruct = make_recording_struct(N, channels, True)

        self.shm = SharedMemory(
            name=name, create=False, size=ctypes.sizeof(cstruct)
        )
        self.data = cstruct.from_buffer(self.shm.buf)
        self.audio = CircularArray(
            np.ndarray(
                (N, channels),
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.data.offset :],
            ),
            SharedInt(self.shm, cstruct.write_counter.offset),
            SharedInt(self.shm, cstruct.counter.offset),
        )
        shared_stft_counter = SharedInt(self.shm, cstruct.stft_counter.offset)
        self.stft = CircularArray(
            np.ndarray(
                (1 + int(config.n_fft / 2), self.N_stft),
                dtype=np.complex64,
                buffer=self.shm.buf[cstruct.stft.offset :],
            ),
            shared_stft_counter,
            axis=-1,
        )
        self.onset_env = CircularArray(
            np.ndarray(
                self.N_stft,
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.onset_env.offset :],
            ),
            shared_stft_counter,
        )
        self.tg = CircularArray(
            np.ndarray(
                (config.tg_win_len, self.N_stft),
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.tg.offset :],
            ),
            shared_stft_counter,
            axis=-1,
        )
        # As these have offset cursors, we don't use CircularArray
        self.mov_max = np.ndarray(
            self.N_stft,
            dtype=np.float32,
            buffer=self.shm.buf[cstruct.mov_max.offset :],
        )
        self.mov_avg = np.ndarray(
            self.N_stft,
            dtype=np.float32,
            buffer=self.shm.buf[cstruct.mov_avg.offset :],
        )
        self.window = sig.windows.hann(config.n_fft).astype(np.float32)
        self.tg_window = sig.windows.hann(config.tg_win_len).astype(np.float32)
        self.onset_env_minmax = EMA_MinMaxTracker(
            min0=0, minmin=0, max0=1, alpha=0.001
        )
        self.logspec_minmax = EMA_MinMaxTracker(
            max0=10, minmax=0, alpha=0.0005
        )
        self.peaks = PeakTracker(self.N_stft, config.onset_det_offset)

    def run(self):
        while not self.data.quit:
            self.do()

    def do(self):
        while not self.data.do_stft:
            if self.data.quit:
                return
            # Currently causes delays
            # time.sleep(self.poll_time)
        self.fft()
        self.data.do_stft = False

    def fft(self):
        stft = np.fft.rfft(self.window * self.audio[-config.n_fft :].mean(-1))
        self.stft.write(stft[:, None], False)
        self.onset_strength()
        self.tempogram()

    def onset_strength(self):
        mag = magsquared(self.stft[-1])
        magm1 = magsquared(self.stft[-2])
        # Convert to DB
        s = 10.0 * np.log10(np.maximum(1e-10, mag))
        self.logspec_minmax.add_sample(s.max())
        s = np.maximum(s, self.logspec_minmax.max_val - 80)
        sm1 = 10.0 * np.log10(np.maximum(1e-10, magm1))
        sm1 = np.maximum(sm1, self.logspec_minmax.max_val - 80)
        # Aggregate frequencies
        onset_env = np.maximum(0.0, s - sm1).mean()

        # Normalize and write
        self.onset_env_minmax.add_sample(onset_env)
        self.onset_env.write(
            np.array([self.onset_env_minmax.normalize_sample(onset_env)]),
            False,
        )
        mov_max_cur = self.onset_env.index_offset(-config.max_offset)
        self.mov_max[mov_max_cur] = np.max(
            self.onset_env[-config.max_length :]
        )
        mov_avg_cur = self.onset_env.index_offset(-config.avg_offset)
        self.mov_avg[mov_avg_cur] = np.mean(
            self.onset_env[-config.avg_length :]
        )

    def tempogram(self):
        tg = np.fft.irfft(
            magsquared(
                np.fft.rfft(
                    self.tg_window * self.onset_env[-config.tg_win_len :],
                    n=config.tg_pad,
                )
            ),
            n=config.tg_pad,
        )[: config.tg_win_len, None]
        # This one increments the shared counter
        self.tg.write(tg / (tg.max() + 1e-10))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("closing")
        del (
            self.data,
            self.audio,
            self.tg,
            self.onset_env,
            self.stft,
            self.mov_avg,
            self.mov_max,
        )
        self.shm.close()
        print("closed")
        self.shm.close()


class RecA(RecAnalysis):
    def do(self):
        while self.data.analysis_action == 0:
            if self.data.quit:
                return
            time.sleep(self.poll_time)
        # TODO: store and pass along start and end events through shm
        match self.data.analysis_action:
            case 1:
                print("Doing 1")
                # Sleep for a bit to make sure we account for bad timing
                sd.sleep(200)
                start = -self.audio.frames_since(self.data.recording_start)
                print("onsets:", self.detect_onsets(samples_to_frames(start)))

        self.data.analysis_action = 0

    def detect_onsets(self, start):
        o = -config.onset_det_offset
        onset_env = self.onset_env[start:o]
        mov_max = self.mov_max[start:o]
        mov_avg = self.mov_avg[start:o]
        detections = self.onset_env * (onset_env == mov_max)
        # Then mask out all entries less than the thresholded average
        detections = detections * (detections >= (mov_avg + config.delta))

        # Initialize peaks array, to be filled greedily
        peaks = []

        # Remove onsets which are close together in time
        last_onset = -np.inf
        for i in np.nonzero(detections)[0]:
            # Only report an onset if the "wait" samples was reported
            if i > last_onset + config.wait:
                peaks.append(i)
                # Save last reported onset
                last_onset = i

        return np.array(peaks)

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
        prev_wc = int(self.write_counter)
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

        if increment:
            self.counter += n
        else:
            self.write_counter -= self.write_counter - prev_wc

    def __repr__(self):
        return (
            self.data.__repr__()
            + f"\nN: {self.N}, i: {self.write_counter}, c: {self.counter}"
        )


    def write(self, arr):
        super().write(arr)
        self.fft()


def analysis(N):
    b = RecAnalysis(N, 1)
    b.run()


if __name__ == "__main__":
    wav, sr = sf.read("../data/drums.wav")
    wav = wav.mean(1)

    N = len(wav)

    with RecMain(N, 1) as rec:
        N_stft = int(np.ceil(N / config.hop_length))

        tg = np.ndarray(
            (config.tg_win_len, N_stft),
            dtype=np.float32,
            buffer=rec.shm.buf[rec.cstruct.tg.offset :],
        )
        stft = np.ndarray(
            (1 + int(config.n_fft / 2), N_stft),
            dtype=np.float32,
            buffer=rec.shm.buf[rec.cstruct.stft.offset :],
        )

        ap = mp.Process(target=analysis, args=(N,))
        ap.start()
        for i in range(0, len(wav) // 1, 256):
            rec.ca.write(wav[i : i + 256, None])
            rec.data.do_stft = True
            time.sleep(0.0005)

        # print(f"{tg=}, {tg.any()}, {stft=}")
        plt.imshow(tg)
        plt.savefig("test.png")
    ap.terminate()
