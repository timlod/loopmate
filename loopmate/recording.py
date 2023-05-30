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


def closest_distance(onsets, grid, beat_len):
    dm = distance_matrix(onsets[:, None], grid[:, None])
    return np.mean(np.sort(dm, axis=0)[:2, :].round())


def find_offset(onsets, bpm, sr=48000, x0=0, **kwargs):
    if len(onsets) == 0:
        return 0
    beat_len = sr // (bpm / 60)
    N = np.ceil(onsets[-1] / beat_len)
    # Add subdivision?
    grid = np.arange(0, N * beat_len, beat_len)

    def closure(offset):
        d = closest_distance(onsets + offset, grid, beat_len)
        return d

    res = minimize(closure, x0=x0, **kwargs)
    return int(res.x)


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
    int_type=ctypes.c_int64,
):
    N_stft = int(np.ceil(N / config.hop_length))

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
            # Flag to indicate that analysis thread has completed
            # Potentially just use the integer and have 0 signal not ready
            ("result_ready", ctypes.c_bool),
            # Type of result that can be taken:
            # 9 = finished recording
            # 8 = preliminary recording
            ("result_type", int_type),
            # Result? array?
            ("write_counter", int_type),
            ("counter", int_type),
            ("data", ctypes.c_float * (N * channels)),
            ("stft_counter", int_type),
            (
                "stft",
                ctypes.c_float * (2 * (1 + int(config.n_fft / 2)) * N_stft),
            ),
            ("onset_env_counter", int_type),
            ("onset_env", ctypes.c_float * N_stft),
            ("mov_max", ctypes.c_float * N_stft),
            ("mov_avg", ctypes.c_float * N_stft),
            ("tg_counter", int_type),
            ("tg", ctypes.c_float * config.tg_win_len * N_stft),
            ("analysis_action", int_type),
            ("quit", ctypes.c_bool),
        ]

    return CRecording


class RecAudio:
    def __init__(self, N, channels, name="recording"):
        cstruct = make_recording_struct(N, channels)
        self.cstruct = cstruct
        self.shm = SharedMemory(
            name=name, create=True, size=ctypes.sizeof(cstruct)
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
    def __init__(self, N, channels, name="recording", poll_time=0.0001):
        # TODO: take N from config or also kwarg other config items
        self.poll_time = poll_time

        self.N_stft = int(np.ceil(N / config.hop_length))
        cstruct = make_recording_struct(N, channels)

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
        self.last_counter = int(self.audio.counter)

        stft_counter = SharedInt(self.shm, cstruct.stft_counter.offset)
        onset_env_counter = SharedInt(
            self.shm, cstruct.onset_env_counter.offset
        )
        tg_counter = SharedInt(self.shm, cstruct.tg_counter.offset)
        self.stft = CircularArray(
            np.ndarray(
                (1 + int(config.n_fft / 2), self.N_stft),
                dtype=np.complex64,
                buffer=self.shm.buf[cstruct.stft.offset :],
            ),
            stft_counter,
            axis=-1,
        )
        self.onset_env = CircularArray(
            np.ndarray(
                self.N_stft,
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.onset_env.offset :],
            ),
            onset_env_counter,
        )
        self.tg = CircularArray(
            np.ndarray(
                (config.tg_win_len, self.N_stft),
                dtype=np.float32,
                buffer=self.shm.buf[cstruct.tg.offset :],
            ),
            tg_counter,
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
        while self.data.counter == self.last_counter:
            if self.data.quit:
                return
            # Currently causes delays
            # time.sleep(self.poll_time)
        self.last_counter = self.data.counter
        self.fft()

    def fft(self):
        stft = np.fft.rfft(self.window * self.audio[-config.n_fft :].mean(-1))
        self.stft.write(stft[:, None])
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
            np.array([self.onset_env_minmax.normalize_sample(onset_env)])
        )
        # Decrement offsets by one as we already incremented onset_env
        mov_max_cur = self.onset_env.index_offset(-config.max_offset - 1)
        self.mov_max[mov_max_cur] = np.max(
            self.onset_env[-config.max_length :]
        )
        mov_avg_cur = self.onset_env.index_offset(-config.avg_offset - 1)
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
        print("Exiting RecAnalysis")
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
    def __init__(self, N, channels, name="recording", poll_time=0.0001):
        super().__init__(N, channels, name, poll_time)
        self.tf = tempo_frequencies(
            config.tg_win_len, config.hop_length, sr=config.sr
        )
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

    def quantize_start(self, lenience=config.sr * 0.1):
        # Sleep for a bit to make sure we account for bad timing, as well as
        # the delay from onset detection
        det_delay_s = config.onset_det_offset * config.hop_length / config.sr
        wait_for_ms = 250
        lookaround_samples = int(wait_for_ms / 1000 * config.sr)
        # Wait for wait_for_ms as well as the onset detection delay
        sd.sleep(wait_for_ms + int(det_delay_s * 1000))
        # We want to check recording_start, so the reference is the frames
        # since then
        ref = self.audio.elements_since(self.data.recording_start)
        # We want to get this many samples before the reference as well

        start = ref + lookaround_samples
        start_frames = -samples_to_frames(start, config.hop_length)
        # In practice, the algorithm finds the onset somewhat later, so
        # we decrement by one to be closer to the actual onset
        onsets = self.detect_onsets(start_frames)
        # Currently, it's very possible that the actual press gets an onset
        # detection, even if quit, which will obviously be the closest one to
        # quantize to. Therefore, let's weight by distance from press and size
        # of the peak. subtract one frame after checking height
        onsets = frames_to_samples(
            onsets - samples_to_frames(lookaround_samples, config.hop_length),
            config.hop_length,
        )
        print(f"RECA: onsets (start): {onsets}")
        _, move = self.quantize_onsets(onsets, lookaround_samples, oe)
        print(
            f"RECA: Moving from {self.data.recording_start=}, {move} to {self.data.recording_start + move}!"
        )
        self.data.recording_start += move

    def quantize_onsets(
        self,
        onsets,
        offset,
        onset_envelope,
        lenience=round(config.sr * 0.1),
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
        offset = samples_to_frames(offset, config.hop_length)
        print(f"{offset=}, {onset_envelope.min()=}, {onset_envelope.max()=}")
        for onset in samples_to_frames(onsets, config.hop_length):
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

    def detect_onsets(self, start):
        o = -config.onset_det_offset
        wc = self.onset_env.write_counter
        onset_env = self.onset_env[start:o]
        mov_max = query_circular(self.mov_max, slice(start, o), wc)
        mov_avg = query_circular(self.mov_avg, slice(start, o), wc)
        detections = onset_env * (onset_env == mov_max)
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

    def quantize_end(self):
        ref_start = self.audio.elements_since(self.data.recording_start)
        start_frame = -samples_to_frames(ref_start, config.hop_length)
        ref_end = self.audio.elements_since(self.data.recording_end)
        n = self.data.recording_end - self.data.recording_start
        n_frames = samples_to_frames(n, config.hop_length)
        end_frame = start_frame + n_frames
        if end_frame > 0:
            end_frame = 0
        # print(f"{ref_start=}, {start_frame=}, {ref_end=}, {end_frame=}, {n=}")
        tg = self.tg[start_frame:end_frame]
        onsets = self.detect_onsets(start_frame)
        bpm = self.tempo(tg)[0]
        beat_len = int(config.sr / (bpm / 60))
        offset = find_offset(
            onsets * config.hop_length, bpm, config.sr, method="Powell"
        )
        if abs(offset) > 512:
            print(f"RECA: Predicted {offset / config.sr} miss!")
            # If within 100ms of a subdivision we assume offbeat start:
            # TODO: missing one case here based on direction of difference
            if beat_len / 2 - abs(offset) < 0.1 * config.sr:
                print(f"RECA: Offset changed from {offset} to {beat_len / 2}")
                offset = offset - np.sign(offset) * beat_len / 2

        # Option: Just extrapolate BPM from start in any case
        n_beats = round(n / beat_len)
        print(f"RECA: {bpm=}, {n_beats=}, {start_frame=}, {beat_len=}")
        end = self.data.recording_start + n_beats * beat_len
        self.data.recording_end = end
        while end > self.audio.counter:
            self.data.result_type = 8
        self.data.result_type = 9

    def tempo(self, tg, agg=np.mean):
        # From librosa.feature.rhythm
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


def analysis(N):
    with RecAnalysis(N, 1) as rec:
        rec.run()
    print("done analysis")


def a2(N):
    with AnalysisOnDemand(N, 1) as rec:
        rec.run()
    print("done a2")


if __name__ == "__main__":
    wav, sr = sf.read("../data/drums.wav")
    wav = wav.mean(1)

    N = len(wav)

    with RecAudio(N, 1) as rec:
        N_stft = int(np.ceil(N / config.hop_length))

        tg = np.ndarray(
            (config.tg_win_len, N_stft),
            dtype=np.float32,
            buffer=rec.shm.buf[rec.cstruct.tg.offset :],
        )
        stft = np.ndarray(
            (1 + int(config.n_fft / 2), N_stft),
            dtype=np.complex64,
            buffer=rec.shm.buf[rec.cstruct.stft.offset :],
        )
        ap1 = mp.Process(target=analysis, args=(N,))
        ap2 = mp.Process(target=a2, args=(N,))
        ap1.start()
        ap2.start()
        time.sleep(0.1)
        for i in range(0, len(wav) // 1, config.blocksize):
            # TODO: possibly increment all counters in main thread, or just use
            # the audio counter, to make sure processes are in sync - poll time
            # causes delays currently, therefore removed
            rec.audio.write(wav[i : i + config.blocksize, None])
            # print("writing done")
            if i == 0:
                time.sleep(0.01)
                print("written 0")
            time.sleep(0.0026666)

        rec.data.analysis_action = 1
        rec.data.recording_start = int(rec.data.counter)

        for i in range(0, len(wav) // 1, config.blocksize):
            # TODO: possibly increment all counters in main thread, or just use
            # the audio counter, to make sure processes are in sync - poll time
            # causes delays currently, therefore removed
            rec.audio.write(wav[i : i + config.blocksize, None])
            # print("writing done")
            if i == 0:
                # time.sleep(0.01)
                print("written 0.2")
            time.sleep(0.002666)

        # rec.data.analysis_action = 1
        rec.data.recording_end = rec.data.recording_start
        del tg, stft
        # print(f"{tg=}, {tg.any()}, {stft=}")
        # plt.imshow(tg)
        # plt.savefig("test.png")
        rec.data.quit = True
        time.sleep(0.2)
        ap1.join()
        ap2.join()
        ap1.close()
        ap2.close()
    # rec.shm.close()
    # rec.shm.unlink()
