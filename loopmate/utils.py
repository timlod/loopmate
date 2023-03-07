from dataclasses import dataclass

import numpy as np
from scipy import signal as sig


def resample(x, sr_source, sr_target):
    """Resample signal to target samplerat

    :param x:
    :param sr_source:
    :param sr_target:
    :returns:

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

    def get_metronome(self, sr):
        clave, msr = sf.read("../data/clave.wav", dtype=np.float32)
        if sr != msr:
            clave = resample(clave, msr, sr)
        beat = int(sr / self.bps)
        out = np.zeros(int(beat * self.beats), dtype=np.float32)
        for i in range(self.beats):
            out[i * beat : i * beat + len(clave)] = clave
        return out


class StreamTime:
    def __init__(self, time, frame):
        self.frame = frame
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
        return f"StreamTime({self.current}, {self.input}, {self.output})"
