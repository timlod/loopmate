import ctypes
import multiprocessing as mp
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal as sig
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

from loopmate import config
from loopmate.circular_array import CircularArray, query_circular
from loopmate.utils import (
    EMA_MinMaxTracker,
    PeakTracker,
    SharedInt,
    StreamTime,
    frames_to_samples,
    magsquared,
    samples_to_frames,
    tempo_frequencies,
)

# TODO: use of config constants not yet consistent - sometimes the kwargs
# referring to those are used, other times not.


def closest_distance(onsets: np.ndarray, grid: np.ndarray) -> float:
    """For each onset/grid pair, compute the distances to the closest two grid
    positions for each onset. Return the mean across all those distances.

    :param onsets: array containing integer indexes corresponding to onsets
    :param grid: array containing beat grid
    """
    dm = distance_matrix(onsets[:, None], grid[:, None])
    return np.mean(np.sort(dm, axis=0)[:2, :].round())


def find_offset(
    onsets: np.ndarray, bpm: float, sr: int = 48000, x0: float = 0.0, **kwargs
) -> int:
    """Given a set of onsets and a BPM, compute an offset which aligns the
    onsets within a grid corresponding to the BPM.

    For example, onsets which are on the offbeat, would return an offset
    corresponding to half a beat given the sample rate.

    :param onsets: array containing integer indexes corresponding to onsets
    :param bpm: estimated BPM of the onsets
    :param sr: sample rate
    :param x0: initial offset guess
    """
    if len(onsets) == 0:
        return 0
    beat_len = sr // (bpm / 60)
    N = np.ceil(onsets[-1] / beat_len)
    # Add subdivision?
    grid = np.arange(0, N * beat_len, beat_len)

    def closure(offset):
        d = closest_distance(onsets + offset, grid)
        return d

    res = minimize(closure, x0=x0, **kwargs)
    return int(res.x)


def make_recording_struct(
    n=config.REC_N,
    channels=config.CHANNELS,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    tg_win_length=config.TG_WIN_LENGTH,
    int_type=ctypes.c_int64,
):
    """Create a ctypes.Structure to use with shared memory in IPC.

    :param n: number of audio samples per channel
    :param channels: number of audio channels
    :param n_fft: size of the fft for the stft
    :param hop_length: hop_length of the stft
    :param tg_win_len: window length for the tempogram
    :param int_type: integer ctype to use
    """
    n_stft = int(np.ceil(n / hop_length))

    class CRecording(ctypes.Structure):
        _fields_ = [
            ### Everything before write_counter is used for IPC
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
            # Type of result that can be taken:
            # 9 = finished recording
            # 8 = preliminary recording
            ("result_type", int_type),
            # Result? array?
            ("write_counter", int_type),
            ("counter", int_type),
            ("data", ctypes.c_float * (n * channels)),
            ("stft_counter", int_type),
            (
                "stft",
                ctypes.c_float * (2 * (1 + int(n_fft / 2)) * n_stft),
            ),
            ("onset_env_counter", int_type),
            ("onset_env", ctypes.c_float * n_stft),
            ("mov_max", ctypes.c_float * n_stft),
            ("mov_avg", ctypes.c_float * n_stft),
            ("tg_counter", int_type),
            ("tg", ctypes.c_float * tg_win_length * n_stft),
            ("analysis_action", int_type),
            ("quit", ctypes.c_bool),
        ]

    return CRecording


class RecAudio:
    """
    Class used to share recorded audio and auxiliary arrays with other
    processes.  Creates shared memory object.  Should be called first in the
    main thread.
    """

    def __init__(
        self, n=config.REC_N, channels=config.CHANNELS, name="recording"
    ):
        cstruct = make_recording_struct(n, channels)
        self.cstruct = cstruct
        self.shm = SharedMemory(
            name=name, create=True, size=ctypes.sizeof(cstruct)
        )
        self.data = cstruct.from_buffer(self.shm.buf)
        self.audio = CircularArray(
            np.ndarray(
                (n, channels),
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.data.offset :],
            ),
            SharedInt(self.shm, cstruct.write_counter.offset),
            SharedInt(self.shm, cstruct.counter.offset),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("exiting RecMain")
        # Without this deletion, we'll get a BufferError, even though
        # https://docs.python.org/3/library/multiprocessing.shared_memory.html#multiprocessing.shared_memory.SharedMemory.close
        # states it's unnecessary
        del (self.data, self.audio)
        self.shm.close()
        self.shm.unlink()
        print("Unlinked!")


class RecAnalysis:
    """
    Class to run ongoing analysis in another process/thread.  Accesses shared
    memory and makes circular arrays for various analytical arrays, such as
    live audio, its STFT and, onset envelope and tempogram.  The latter are
    updated whenever the counter inside the shared memory buffer is incremented
    (presumably in the main process).
    """

    def __init__(
        self,
        n=config.REC_N,
        channels=config.CHANNELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        tg_win_length=config.TG_WIN_LENGTH,
        name="recording",
        poll_time=0.0001,
    ):
        # TODO: take N from config or also kwarg other config items
        self.poll_time = poll_time

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_stft = int(np.ceil(n / hop_length))
        self.tg_win_length = tg_win_length
        self.tg_pad = 2 * tg_win_length - 1
        cstruct = make_recording_struct(
            n, channels, n_fft, hop_length, tg_win_length
        )

        self.shm = SharedMemory(
            name=name, create=False, size=ctypes.sizeof(cstruct)
        )
        self.data = cstruct.from_buffer(self.shm.buf)
        self.audio = CircularArray(
            np.ndarray(
                (n, channels),
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.data.offset :],
            ),
            SharedInt(self.shm, cstruct.write_counter.offset),
            SharedInt(self.shm, cstruct.counter.offset),
        )
        self.last_counter = int(self.audio.counter)

        stft_counter = SharedInt(self.shm, cstruct.stft_counter.offset)
        onset_env_counter = SharedInt(
            self.shm, cstruct.onset_env_counter.offset
        )
        tg_counter = SharedInt(self.shm, cstruct.tg_counter.offset)
        self.stft = CircularArray(
            np.ndarray(
                (1 + int(n_fft / 2), self.n_stft),
                dtype=np.complex64,
                buffer=self.shm.buf[cstruct.stft.offset :],
            ),
            stft_counter,
            axis=-1,
        )
        self.onset_env = CircularArray(
            np.ndarray(
                self.n_stft,
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.onset_env.offset :],
            ),
            onset_env_counter,
        )
        self.tg = CircularArray(
            np.ndarray(
                (tg_win_length, self.n_stft),
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.tg.offset :],
            ),
            tg_counter,
            axis=-1,
        )
        # As these have offset cursors, we don't use CircularArray
        self.mov_max = np.ndarray(
            self.n_stft,
            dtype=np.float32,
            buffer=self.shm.buf[cstruct.mov_max.offset :],
        )
        self.mov_avg = np.ndarray(
            self.n_stft,
            dtype=np.float32,
            buffer=self.shm.buf[cstruct.mov_avg.offset :],
        )
        self.window = sig.windows.hann(n_fft).astype(np.float32)
        self.tg_window = sig.windows.hann(tg_win_length).astype(np.float32)
        self.onset_env_minmax = EMA_MinMaxTracker(
            min0=0, minmin=0, max0=1, alpha=0.001
        )
        self.logspec_minmax = EMA_MinMaxTracker(
            max0=10, minmax=0, alpha=0.0005
        )

    def run(self):
        while not self.data.quit:
            self.do()

    def do(self):
        while self.data.counter == self.last_counter:
            if self.data.quit:
                return
            # Currently causes delays
            # time.sleep(self.poll_time)
        self.last_counter = self.data.counter
        self.fft()

    def fft(self):
        """Computes a single frame of the STFT of the most recent audio, as
        well as its onset strength and tempogram frame.
        """
        stft = np.fft.rfft(self.window * self.audio[-self.n_fft :].mean(-1))
        self.stft.write(stft[:, None])
        self.onset_strength()
        self.tempogram()

    def onset_strength(self):
        """
        Compute onset envelope and auxiliary (max and average) arrays of most
        recent STFT frames.
        """
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
            np.array([self.onset_env_minmax.normalize_sample(onset_env)])
        )
        # Decrement offsets by one as we already incremented onset_env
        mov_max_cur = self.onset_env.index_offset(-config.MAX_OFFSET - 1)
        self.mov_max[mov_max_cur] = np.max(
            self.onset_env[-config.MAX_LENGTH :]
        )
        mov_avg_cur = self.onset_env.index_offset(-config.AVG_OFFSET - 1)
        self.mov_avg[mov_avg_cur] = np.mean(
            self.onset_env[-config.AVG_LENGTH :]
        )

    def tempogram(self):
        """
        Compute tempogram given the most recent onset_envelope frames.
        """
        tg = np.fft.irfft(
            magsquared(
                np.fft.rfft(
                    self.tg_window * self.onset_env[-self.tg_win_length :],
                    n=self.tg_pad,
                )
            ),
            n=self.tg_pad,
        )[: self.tg_win_length, None]
        # This one increments the shared counter
        self.tg.write(tg / (tg.max() + 1e-10))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting RecAnalysis")
        # Delete all shared memory objects. This shouldn't be necessary, but
        # for some reason they're not cleared automatically.
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
        print("Closed RecAnalysis SHM")


class AnalysisOnDemand(RecAnalysis):
    """
    Class to compute onset detection and quantization on demand.  These actions
    can be requested/signalled in another process using data.analysis_action.
    """

    def __init__(
        self,
        n=config.REC_N,
        channels=config.CHANNELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        tg_win_length=config.TG_WIN_LENGTH,
        name="recording",
        poll_time=0.0001,
    ):
        super().__init__(
            n, channels, n_fft, hop_length, tg_win_length, name, poll_time
        )
        self.tf = tempo_frequencies(tg_win_length, hop_length, sr=config.SR)
        self.bpm_logprior = (
            -0.5 * ((np.log2(self.tf) - np.log2(100)) / 1.0) ** 2
        )[:, None]

    def do(self):
        while self.data.analysis_action == 0:
            if self.data.quit:
                return
            # time.sleep(self.poll_time)
        # TODO: store and pass along start and end events through shm
        match self.data.analysis_action:
            case 1:
                print("RECA: Quantize start")
                self.quantize_start()
            case 2:
                print("RECA: Quantize end")
                self.quantize_end()

        self.data.analysis_action = 0

    def detect_onsets(self, start: int):
        """Detect onsets since requested start frame based on shared memory
        arrays.

        Online implementation of librosa's onset_detect algorithm (as of
        0.10.0).

        :param start: start frame indexing onset envelope.  Should be a
            negative index.
        """
        o = -config.ONSET_DET_OFFSET
        wc = self.onset_env.write_counter
        onset_env = self.onset_env[start:o]
        mov_max = query_circular(self.mov_max, slice(start, o), wc)
        mov_avg = query_circular(self.mov_avg, slice(start, o), wc)
        detections = onset_env * (onset_env == mov_max)
        # Then mask out all entries less than the thresholded average
        detections = detections * (detections >= (mov_avg + config.DELTA))

        # Initialize peaks array, to be filled greedily
        peaks = []

        # Remove onsets which are close together in time
        last_onset = -np.inf
        for i in np.nonzero(detections)[0]:
            # Only report an onset if the "wait" samples was reported
            if i > last_onset + config.WAIT:
                peaks.append(i)
                # Save last reported onset
                last_onset = i

        return np.array(peaks), onset_env

    def quantize_onsets(
        self,
        onsets,
        offset,
        onset_envelope,
        lenience=round(config.SR * 0.1),
        strength_weight=0.5,
        window_size=5,
    ) -> (int, int):
        """
        Quantize recording marker to onsets if within some interval from them.
        Also returns the difference between original frame and quantized frame.
        Weights the onsets by distance from event (assumed to be 0) and onset
        strength.  This is done such that a quiet onset generated by hitting
        the MIDI pad (which should always be close to 0) is not chosen over a
        strong onset, for example by the bass drum on the 1, a little farther
        away.

        To use only distances, pass strength_weight = 0.

        :param onsets: array of detected onsets, with 0 indicating the event
            happening at the offset
        :param offset: offset at which the reference frame starts in
            onset_envelope
        :param onset_envelope: array of the onset envelope used to detect
            onsets
        :param lenience: quantize if within this many samples from detected
            onsets
        :param strength_weight: weight for onset strength in the decision
            making
        :param window_size: size of the window around the detected onsets on
            the envelope to check for onset strength
        """
        if len(onsets) == 0:
            return 0, 0

        # Find the strengths of the onsets in the onset envelope
        strengths = []
        offset = samples_to_frames(offset, self.hop_length)
        for onset in samples_to_frames(onsets, self.hop_length):
            start = max(0, offset + onset - window_size)
            end = min(len(onset_envelope), offset + onset + window_size)
            strengths.append(np.max(onset_envelope[start:end]))

        strengths = np.array(strengths)

        # Calculate the distances between frame and onsets
        distances = np.abs(onsets)

        # Calculate the geometric mean of distance and strength weighted by
        # strength_weight.
        weighted_distances = (
            distances ** (1 - strength_weight)
            * (1 - strengths) ** strength_weight
        )
        # weighted_distances = distances * (1 - strengths) * strength_weight

        # Find the onset with the minimum weighted distance
        if distances[(i := weighted_distances.argmin())] < lenience:
            move = onsets[i]
        else:
            move = 0

        return move, move

    def quantize_start(self, lenience=config.SR * 0.1):
        # Sleep for a bit to make sure we account for bad timing, as well as
        # the delay from onset detection
        det_delay_s = config.ONSET_DET_OFFSET * self.hop_length / config.SR
        wait_for_ms = 250
        lookaround_samples = int(wait_for_ms / 1000 * config.SR)
        # Wait for wait_for_ms as well as the onset detection delay
        sd.sleep(wait_for_ms + int(det_delay_s * 1000))
        # We want to check recording_start, so the reference is the frames
        # since then
        ref = self.audio.elements_since(self.data.recording_start)
        # We want to get this many samples before the reference as well

        start = ref + lookaround_samples
        start_frames = -samples_to_frames(start, self.hop_length)
        # In practice, the algorithm finds the onset somewhat later, so
        # we decrement by one to be closer to the actual onset
        onsets, onset_envelope = self.detect_onsets(start_frames)
        # Currently, it's very possible that the actual press gets an onset
        # detection, even if quit, which will obviously be the closest one to
        # quantize to. Therefore, let's weight by distance from press and size
        # of the peak. subtract one frame after checking height
        onsets = frames_to_samples(
            onsets - samples_to_frames(lookaround_samples, self.hop_length),
            self.hop_length,
        )
        print(f"RECA: onsets (start): {onsets}")
        _, move = self.quantize_onsets(
            onsets, lookaround_samples, onset_envelope
        )
        print(
            f"RECA: Moving from {self.data.recording_start=}, {move} to {self.data.recording_start + move}!"
        )
        self.data.recording_start += move

    def quantize_end(self):
        """Quantize the end marker of recorded audio.  Uses detected onsets to
        compute BPM and does one of the following:

            - quantize to closest strong onset next to the end marker

            - in absence of onset, quantize to closest beat marker extrapolated
              from start and the estimated BPM
        """
        ref_start = self.audio.elements_since(self.data.recording_start)
        start_frame = -samples_to_frames(ref_start, self.hop_length)
        n = self.data.recording_end - self.data.recording_start
        n_frames = samples_to_frames(n, self.hop_length)
        end_frame = start_frame + n_frames
        if end_frame > 0:
            end_frame = 0

        tg = self.tg[start_frame:end_frame]
        onsets, onset_envelope = self.detect_onsets(start_frame)

        bpm = self.tempo(tg)[0]
        beat_len = int(config.SR / (bpm / 60))
        offset = find_offset(
            onsets * self.hop_length, bpm, config.SR, method="Powell"
        )
        if abs(offset) > 512:
            print(f"RECA: Predicted {offset / config.SR} miss!")
            # If within 100ms of a subdivision we assume offbeat start:
            # TODO: missing one case here based on direction of difference
            if beat_len / 2 - abs(offset) < 0.1 * config.SR:
                print(f"RECA: Offset changed from {offset} to {beat_len / 2}")
                offset = offset - np.sign(offset) * beat_len / 2

        # Option: Just extrapolate BPM from start in any case
        n_beats = round(n / beat_len)
        print(f"RECA: {bpm=}, {n_beats=}, {start_frame=}, {beat_len=}")
        end = self.data.recording_start + n_beats * beat_len
        self.data.recording_end = end
        self.data.result_type = 8

    def tempo(self, tg, agg=np.mean) -> float:
        """Compute BPM estimate.

        Taken from librosa.feature.rhythm:
        Copyright (c) 2013--2023, librosa development team.

        :param tg: tempogram slice to estimate the BPM for
        :param agg: aggregation method - if None, return estimate for each
            frame of the tempogram.
        """
        best_period = np.argmax(
            np.log1p(1e6 * tg) + self.bpm_logprior, axis=-2
        )
        if agg is not None:
            tg = agg(tg, axis=-1, keepdims=True)
        best_period = np.argmax(
            np.log1p(1e6 * tg) + self.bpm_logprior, axis=-2
        )
        return np.take(self.tf, best_period)

    def bpm_quantize_start(self):
        pass

    def get_last_impulse(self):
        pass
