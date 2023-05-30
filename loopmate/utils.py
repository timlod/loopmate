import ctypes
from collections import deque
from dataclasses import dataclass

# import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import soundfile as sf
from scipy import signal as sig

CLAVE, MSR = sf.read("../data/clave.wav", dtype=np.float32)
CLAVE = CLAVE[:, None]


def frames_to_samples(frames: np.ndarray, hop_length: int = 256) -> np.ndarray:
    """Convert frame index to sample index.

    :param frames: array containing frame indices.
    :param hop_length: hop_length of the STFT
    """
    return frames * hop_length


def samples_to_frames(
    samples: np.ndarray, hop_length: int = 256
) -> np.ndarray:
    return samples // hop_length


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
        # Make room for the next channel number (8 bits)
        result <<= 8
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
        # Extract the least significant 8 bits
        channel = value & 0xFF
        channels.insert(0, channel - 1)
        # Shift the value to the right by 8 bits
        value >>= 8
    return channels


def resample(x: np.ndarray, sr_source: int, sr_target: int):
    """Resample signal to target samplerate

    :param x: audio file to resample of shape (n_samples, channels)
    :param sr_source: samplerate of x
    :param sr_target: target samplerate
    """
    c = sr_target / sr_source
    n_target = int(np.round(len(x) * c))
    return sig.resample(x, n_target)


def tempo_frequencies(n_bins: int, hop_length: int = 256, sr: int = 48000):
    # From librosa.convert
    bin_frequencies = np.zeros(int(n_bins), dtype=np.float64)

    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, n_bins))

    return bin_frequencies


@dataclass
class Metre:
    bpm: float
    beats: int
    divisions: int

    @property
    def bps(self):
        return self.bpm / 60

    def get_metronome(self, sr: int):
        if sr != MSR:
            clave = resample(CLAVE, MSR, sr)
        else:
            clave = CLAVE
        beat = int(sr / self.bps)
        out = np.zeros((int(beat * self.beats), 1), dtype=np.float32)
        for i in range(self.beats):
            out[i * beat : i * beat + len(clave)] = clave
        return out


class StreamTime:
    # TODO: may precompute delays to save some precious cycles
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


class EMA_MinMaxTracker:
    def __init__(
        self,
        alpha=0.0001,
        eps=1e-10,
        min0=0.0,
        max0=-np.inf,
        minmin=-np.inf,
        minmax=-np.inf,
    ):
        self.alpha = np.float32(alpha)
        self.ialpha = np.float32(1.0 - alpha)
        self.eps = np.float32(eps)
        self.min_val = np.float32(min0)
        self.max_val = np.float32(max0)
        self.minmin = np.float32(minmin)
        self.minmax = np.float32(minmax)

    def add_sample(self, sample):
        # Update min_val and max_val using exponential moving average
        # sample = abs(sample)
        if sample < self.minmin:
            self.min_val = self.minmin
        elif sample < self.min_val:
            self.min_val = sample
        else:
            self.min_val = self.min_val * self.ialpha + sample * self.alpha

        if sample < self.minmax:
            self.max_val = self.minmax
        elif sample > self.max_val:
            self.max_val = sample
        else:
            self.max_val = self.max_val * self.ialpha + sample * self.alpha

    def normalize_sample(self, sample):
        sample -= self.min_val
        return sample / (self.max_val + self.eps)


class PeakTracker:
    def __init__(self, N, offset=0):
        self.N = N
        self.absolute = deque()
        self.current_step = 0
        self.offset = offset

    def add_element(self):
        self.absolute.append(self.current_step - self.offset)

    def step(self):
        self.current_step += 1
        while self.absolute and self.absolute[0] - self.N > self.current_step:
            self.absolute.popleft()

    @property
    def last(self):
        if self.absolute:
            return self.absolute[-1] - self.current_step
        else:
            # Sufficiently large negative number indicates there's no last peak
            return -100000

    @property
    def peaks(self):
        return np.array(self.absolute) - self.current_step


def magsquared(x):
    return x.real**2 + x.imag**2


def tempo(
    tg: np.ndarray,
    hop_length: int = 256,
    win_length: int = 384,
    start_bpm: int = 120,
    std_bpm: float = 1.0,
    agg=np.mean,
    sr=48000,
):
    # From librosa.feature.rhythm
    if agg is not None:
        tg = agg(tg, axis=-1, keepdims=True)
    bpms = tempo_frequencies(win_length, hop_length, sr=sr)
    logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2
    logprior = logprior[:, None]
    best_period = np.argmax(np.log1p(1e6 * tg) + logprior, axis=-2)
    return np.take(bpms, best_period)


def get_bpm(groups):
    amps = [sum(peak[0] for peak in group) for group in groups]
    return groups[np.argmax(amps)][-1][-1]


def estimate_bpm(x, sr, tolerance=0.01, ds=1, convolve=True):
    if x.ndim > 1:
        x = x.mean(axis=1)

    if convolve:
        kernel_size = round(sr * 0.001)
        kernel = np.ones(kernel_size) / kernel_size
        x = np.convolve(x**2, kernel, mode="same")

    # Normalize by RMS
    norm = np.sqrt(np.mean(x**2))
    x = x / norm

    # Calculate the FFT and obtain the magnitude spectrum
    spectrum = np.abs(np.fft.rfft(x[::ds], n=len(x)))

    # Calculate the frequency corresponding to each FFT bin
    freqs = np.fft.rfftfreq(x.size, 1 / (sr / ds))
    # Convert frequencies to BPM
    bpms = 60 * freqs

    # Filter out BPMs that are outside the reasonable range (e.g., 50-200 BPM)
    valid_indices = (bpms >= 30) & (bpms <= 300)
    valid_bpms = bpms[valid_indices]
    valid_spectrum = spectrum[valid_indices]

    plt.plot(valid_bpms, valid_spectrum)

    # Use top n frequencies to allow
    peaks = sp.signal.find_peaks(valid_spectrum)[0]
    groups = []

    for peak in peaks:
        found_group = False
        amplitude = valid_spectrum[peak]
        bpm = valid_bpms[peak]
        for group in groups:
            for _, _, _, group_bpm in group:
                ratio = bpm / group_bpm
                dev = abs(round(ratio) - ratio)
                if dev <= tolerance:
                    group.append(
                        (
                            amplitude,
                            dev,
                            round(ratio),
                            valid_bpms[peak],
                        )
                    )
                    found_group = True
                    break
            if found_group:
                break

        if not found_group:
            groups.append([(amplitude, 0, 1, valid_bpms[peak])])
    return groups, get_bpm(groups)


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

    def __int__(self):
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
