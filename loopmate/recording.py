from __future__ import annotations

import numpy as np
import sounddevice as sd
from scipy import signal as sig

from loopmate import config
from loopmate.loop import Audio
from loopmate.utils import CircularArray, StreamTime

RAMP = np.linspace(1, 0, config.blend_frames, dtype=np.float32)[:, None]
POP_WINDOW = sig.windows.hann(config.blend_frames)[:, None]
# TODO: If there are added/dropped frames during recording, may have to
# account for them


class Recording:
    def __init__(
        self,
        rec: CircularArray,
        callback_time: StreamTime,
        start_time: float,
        loop_length: int | None = None,
    ):
        # Between pressing and the time the last callback happened are this
        # many frames
        frames_since = round(callback_time.timediff(start_time) * config.sr)

        # Our reference will be the the frame indata_at which a press was
        # registered, which is the frame indata_at callback time plus the
        # frames elapsed since and the (negative) output delay, which
        # represents difference in time between buffering a sample and the
        # sample being passed into the DAC
        if loop_length is None:
            reference_frame = 0
        else:
            reference_frame = (
                callback_time.frame
                + frames_since
                + round(callback_time.output_delay * config.sr)
            )

        self.rec_start = (
            rec.counter
            + frames_since
            + round(callback_time.input_delay * config.sr)
        )

        # Quantize to loop_length if it exists
        if loop_length is not None:
            self.loop_length = loop_length
            if reference_frame > loop_length:
                reference_frame -= loop_length
            self.start_frame, move = self.quantize(reference_frame)
            print(
                f"\n\rMoving {move} from {reference_frame} to {self.start_frame}"
            )
            self.rec_start += move
        else:
            self.start_frame = 0
            self.loop_length = None

        self.rec = rec

    def finish(self, t, callback_time):
        # Between pressing and the time the last callback happened are this
        # many frames
        frames_since = round(callback_time.timediff(t) * config.sr)
        self.rec_stop = (
            self.rec.counter
            + frames_since
            + round(callback_time.input_delay * config.sr)
        )

        n = self.rec_stop - self.rec_start
        if self.loop_length is not None:
            reference_frame = self.start_frame + n
            self.end_frame, move = self.quantize(reference_frame, False)
            print(f"\n\rMove {move} to {self.end_frame}")
            self.rec_stop += move
            n += move
        else:
            self.end_frame = self.loop_length = n

        if self.rec_stop > self.rec.counter:
            wait_for = (self.rec_counter - self.rec_stop) / config.sr
            # Need to specify sleep time in ms, add blocksize to make sure we
            # don't get a block too few
            sd.sleep(int((wait_for + config.blocksize) * 1000))
            assert self.rec_stop >= self.rec.counter

        back = self.rec.frames_since(self.rec_stop)
        rec_i = -(n + back)
        recording = self.rec[rec_i:-back]
        self.antipop(recording, self.rec[rec_i - config.blend_frames : rec_i])

        # We need to add the starting frame (in case we start this audio late)
        # as well as subtract the audio delay we added when we started
        # recording
        n_loop_iter = int(2 * np.ceil(np.log2(n / self.loop_length)))
        n += self.start_frame - round(callback_time.output_delay * config.sr)
        if n > n_loop_iter * self.loop_length:
            n = n % self.loop_length
        audio = Audio(recording, self.loop_length, n)
        return audio

    def antipop(self, recording, xfade_end):
        # If we have a full loop, blend from pre-recording, else 0 blend
        if (self.end_frame % self.loop_length) == 0:
            recording[-config.blend_frames :] = (
                RAMP * recording[-config.blend_frames :]
                + (1 - RAMP) * xfade_end
            )
        else:
            n_pw = len(POP_WINDOW) // 2
            recording[:n_pw] *= POP_WINDOW[:n_pw]
            recording[-n_pw:] *= POP_WINDOW[-n_pw:]

    def quantize(self, frame, start=True, lenience=0.2) -> (int, int):
        """Quantize start or end recording marker to the loop boundary if
        within some interval from them.  Also returns difference between
        original frame and quantized frame.

        :param frame: start or end recording marker
        :param start: True for start, or False for end
        :param lenience: quantize if within this many seconds from the loop
            boundary

            For example, for sr=48000, the lenience (indata_at 200ms) is 9600
            samples.  If the end marker is indata_at between 38400 and 57600,
            it will instead be set to 48000, the full loop.
        """
        loop_n, frame_rem = np.divmod(frame, self.loop_length)
        lenience = config.sr * lenience
        if start:
            if frame < lenience:
                return 0, -frame
            elif frame > (self.loop_length - lenience):
                return 0, self.loop_length - frame
            else:
                return frame, 0
        else:
            if frame_rem < lenience:
                return loop_n * self.loop_length, -frame_rem
            elif frame_rem > (self.loop_length - lenience):
                return (
                    loop_n + 1
                ) * self.loop_length, self.loop_length - frame_rem
            else:
                return frame, 0
