from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Union

# import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig

from loopmate.utils import (
    EMA_MinMaxTracker,
    PeakTracker,
    SharedInt,
    magsquared,
    samples_to_frames,
    tempo,
)


def query_circular(
    data: np.ndarray,
    idx_slice: slice,
    counter: int,
    out: Optional[np.ndarray] = None,
    axis: int = 0,
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
    :param axis: either 0 (slice first axis) or -1 (slice last axis)
    """
    assert isinstance(
        idx_slice, slice
    ), f"Use slice for indexing! (Got {idx_slice})"
    start, stop = idx_slice.start or 0, idx_slice.stop or 0
    N = data.shape[axis]
    assert (
        -N < start < stop <= 0
    ), f"Can only slice at most N ({N}) items backward on!"
    l_i = counter + start
    r_i = counter + stop
    # print(f"{counter=}, {start=}, {stop=}")

    if l_i < 0 <= r_i:
        if axis != 0:
            return np.concatenate(
                (data[..., l_i:], data[..., :r_i]), out=out, axis=axis
            )
        else:
            return np.concatenate((data[l_i:], data[:r_i]), out=out, axis=axis)
    else:
        if out is not None:
            if axis != 0:
                out[:] = data[..., l_i:r_i]
            else:
                out[:] = data[l_i:r_i]
            return out
        else:
            if axis != 0:
                return data[..., l_i:r_i].copy()
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
        return query_circular(self.data, i, int(self.write_counter), out)

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
        # print(f"{l_i=}, {self.write_counter=}, {arr_i=}")
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


class CircularArraySTFT(CircularArray):
    def __init__(self, N, channels=2, n_fft=2048, hop_length=256, sr=44100):
        super().__init__(N, channels)
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

    def index_offset_stft(self, offset: Union[int, np.ndarray]):
        if isinstance(offset, np.ndarray):
            i = self.stft_counter + offset
            return np.where(
                i > self.N_stft,
                i % self.N_stft,
                np.where(i < 0, self.N_stft + i, i),
            )
        else:
            if (i := self.stft_counter + offset) > self.N_stft:
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
        bpm = tempo(tg, hop_length=self.hop_length, win_length=self.tg_win_len)
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
