from dataclasses import dataclass

import numpy as np


@dataclass
class FoR:
    # sample referring to the index anchored at the start of the last recorded
    # buffer
    reference_frame: int
    # actual array index of reference_frame
    at: int
    # loop_length, should be 0 if not existing
    n: int

    # given rf and fb, return index to split array on

    def __post_init__(self):
        pass

    def org(self, frames_before):
        return self.at - frames_before

    def fr(self, frames_before):
        i = self.reference_frame - frames_before
        if i < 0:
            return self.n + i
        else:
            return i

    def change_at(self, at):
        # Should change to frames_before
        self.at = at


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
