from __future__ import annotations

import queue
from collections import deque
from dataclasses import dataclass, field
from warnings import warn

import numpy as np
import sounddevice as sd
from scipy import signal as sig

from loopmate import config
from loopmate.actions import Actions, Sample, Start, Stop
from loopmate.circular_array import CircularArray
from loopmate.utils import CLAVE, StreamTime, channels_to_int

RAMP = np.linspace(1, 0, config.blend_frames, dtype=np.float32)[:, None]
POP_WINDOW = sig.windows.hann(config.blend_frames)[:, None]

# TODO: BPM sync & metronome anchor


@dataclass
class Audio:
    # store info about different audio files to loop, perhaps with actions
    # around them as well, i.e. meter, bpm, etc.
    audio: np.ndarray = field(repr=False)
    loop_length: int | None = None
    pos_start: int = 0
    current_sample: int = 0
    n: int = field(init=False)
    channels: int = field(init=False)

    def __post_init__(self):
        try:
            self.n, self.channels = self.audio.shape
        except ValueError:
            self.n, self.channels = len(self.audio), 1
            self.audio = self.audio[:, None]
        if self.loop_length is None:
            self.loop_length = self.n

        left = self.pos_start
        self.n += left
        if self.n <= self.loop_length:
            right = self.loop_length - self.n
            self.n_loop_iter = 1
        else:
            # Need to zero-pad right to a power of 2 amount of loop iterations
            self.n_loop_iter = int(
                2 ** np.ceil(np.log2(self.n / self.loop_length))
            )

            right = self.n_loop_iter * self.loop_length - self.n

        self.audio = np.concatenate(
            [
                np.zeros((left, self.channels), dtype=np.float32),
                self.audio,
                np.zeros((right, self.channels), dtype=np.float32),
            ]
        )
        self.n += right
        self._audio = self.audio.copy()

        self.actions = Actions(self)

    def get_n(self, samples: int):
        """Return the next batch of audio in the loop

        :param frames: number of audio samples to return
        """
        leftover = self.n - self.current_sample
        chunksize = min(leftover, samples)
        current_sample = self.current_sample

        if leftover <= samples:
            out = self.audio[
                np.r_[
                    current_sample : current_sample + chunksize,
                    : samples - leftover,
                ]
            ]
            self.current_sample = samples - leftover
        else:
            out = self.audio[current_sample : current_sample + chunksize]
            self.current_sample += samples

        self.actions.run(out, current_sample, self.current_sample)
        return out

    def reset_audio(self):
        self.audio = self._audio.copy()

    def make_anchor(self):
        self.loop_length = self.loop_length * self.n_loop_iter
        self.n_loop_iter = 1

    def __repr__(self):
        return f"{self.n=}, {self.audio.shape=}, {self.loop_length=}, {self.n_loop_iter=}, {self.current_sample=}"


class Loop:
    def __init__(self, recording, anchor: Audio | None = None):
        self.audios = []
        self.new_audios = queue.Queue()
        self.anchor = anchor
        if anchor is not None:
            self.audios.append(anchor)

        self.rec = recording
        # Always record audio buffers so we can easily look back for loopables
        self.rec_audio = self.rec.audio

        # Global actions applied to fully mixed audio
        self.actions = Actions()

        self.stream = sd.Stream(
            samplerate=config.sr,
            device=config.device,
            channels=config.channels,
            callback=self._get_callback(),
            latency=config.latency,
            blocksize=config.blocksize,
        )
        self.callback_time = None
        self.last_out = deque(maxlen=20)

    def add_track(self, audio):
        if len(self.audios) == 0:
            if not isinstance(audio, Audio):
                audio = Audio(audio)
            self.anchor = audio
            self.audios.append(self.anchor)
        else:
            if not isinstance(audio, Audio):
                audio = Audio(audio, self.anchor.loop_length)
            audio.actions.append(
                Start(audio.current_sample, audio.loop_length)
            )
            self.new_audios.put(audio)
            self.audios.append(audio)

    def _get_callback(self):
        """
        Creates callback function for this loop.
        """

        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)

            # If no audio is present
            if self.anchor is None:
                current_sample = 0
            else:
                current_sample = self.anchor.current_sample

            # Align current frame for newly put audio - maybe not necessary now
            for i in range(self.new_audios.qsize()):
                audio = self.new_audios.get()
                current_loop_iter = audio.current_sample // audio.loop_length
                audio.current_sample = (
                    current_loop_iter * audio.loop_length + current_sample
                )

            # These times/frame refer to the frame that is processed in this
            # callback
            self.callback_time = StreamTime(time, current_sample, frames)

            # Copy necessary as indata arg is passed by reference
            indata = indata.copy()
            self.rec_audio.write(indata)

            outdata[:] = 0.0
            for audio in self.audios:
                a = audio.get_n(frames)
                outdata[:] += a

            # Store last output buffer to potentially send a slightly delayed
            # version to headphones (to match the speaker sound latency). We do
            # this before running actions such that we can mute the two
            # separately
            self.last_out.append((self.callback_time, outdata.copy()))

            next_frame = (
                0 if self.anchor is None else self.anchor.current_sample
            )
            self.actions.run(outdata, current_sample, next_frame)

        return callback

    def start(self):
        self.stream.stop()
        # self.current_sample = 0
        if self.anchor is not None:
            self.actions.actions.append(
                Start(self.anchor.current_sample, self.anchor.loop_length)
            )
        self.stream.start()

    def stop(self):
        if self.anchor is not None:
            print(f"Stopping stream action. {self.anchor.current_sample}")
            # self.actions.actions.append(
            #     Stop(self.anchor.current_sample, self.anchor.loop_length)
            # )
            self.stream.stop()

    def event_counter(self) -> (int, int):
        """Return the recording counter location corresponding to the time when
        this function was called, as well as the offset samples relative to the
        beginning of the current audio block/frame.
        """
        t = self.stream.time
        samples_since = round(self.callback_time.timediff(t) * config.sr)
        return (
            self.rec_audio.counter
            + samples_since
            + round(self.callback_time.input_delay * config.sr)
        ), samples_since

    def startrec(self, lenience=config.sr * 0.2, channels=[0, 1]):
        # recording_start is the counter in rec_audio coinciding with the
        # event, start_sample is the sample number within the loop the event
        # coincides with. For a new loop this will be 0.
        self.rec.data.recording_start, samples_since = self.event_counter()

        if self.anchor is not None:
            start_sample = (
                self.callback_time.frame
                + samples_since
                + round(self.callback_time.output_delay * config.sr)
            )
            if start_sample > self.anchor.loop_length:
                start_sample -= self.anchor.loop_length
            self.start_sample, move = quantize(
                start_sample, self.anchor.loop_length, lenience=lenience
            )
            self.rec.data.recording_start += move
        else:
            # Initiate quantize_start in AnalysisOnDemand
            self.rec.data.analysis_action = 1
            self.rec.data.channels = channels_to_int(channels)
            self.start_sample = 0

        print(f"Load: {100 * self.stream.cpu_load:.2f}%")

    def stoprec(self, lenience=config.sr * 0.2):
        self.rec.data.recording_end, _ = self.event_counter()
        N = self.rec.data.recording_end - self.rec.data.recording_start

        if self.anchor is not None:
            loop_length = self.anchor.loop_length
            end_sample = self.start_sample + N
            end_sample, move = quantize(
                end_sample, loop_length, False, lenience
            )
            self.rec.data.recording_end += move
            N += move
            if N == 0:
                warn("Loop so short it quantized to 0.")
                return
        else:
            # Initiate quantize_end in AnalysisOnDemand
            self.rec.data.analysis_action = 3
            while self.rec.data.result_type < 8:
                # Waiting for end quantization to finish
                sd.sleep(0)
            N = self.rec.data.recording_end - self.rec.data.recording_start
            end_sample = loop_length = N

        start_back = -self.rec_audio.elements_since(
            self.rec.data.recording_start
        )
        rec = self.rec_audio[start_back:][:N]
        n = N
        n_loop_iter = int(2 ** np.ceil(np.log2(n / loop_length)))

        n += self.start_sample - round(
            self.callback_time.output_delay * config.sr
        )
        if n > n_loop_iter * loop_length:
            n = n % loop_length
        audio = Audio(rec, loop_length, self.start_sample, current_sample=n)
        self.add_track(audio)

        while self.rec.data.recording_end > self.rec_audio.counter:
            # Maybe make 0
            sd.sleep(int(config.blocksize / config.sr * 1000))

        start_back = -self.rec_audio.elements_since(
            self.rec.data.recording_start
        )
        rec = self.rec_audio[start_back:][:N]
        self.antipop(
            rec,
            self.rec_audio[start_back - config.blend_frames : start_back],
            end_sample,
        )
        audio.audio[self.start_sample : self.start_sample + N] = rec
        self.rec.data.result_type = 0

    def antipop(self, audio, xfade_end, end_sample):
        # If we have a full loop, blend from pre-recording, else 0 blend
        if (end_sample % self.anchor.loop_length) == 0:
            audio[-config.blend_frames :] = (
                RAMP * audio[-config.blend_frames :] + (1 - RAMP) * xfade_end
            )
        else:
            n_pw = len(POP_WINDOW) // 2
            audio[:n_pw] *= POP_WINDOW[:n_pw]
            audio[-n_pw:] *= POP_WINDOW[-n_pw:]

    def backcapture(self, n):
        print(f"Backcapture {n=}!")
        self.startrec(self.anchor.loop_length // 2)
        self.rec.data.recording_start -= self.anchor.loop_length * n
        self.stoprec(self.anchor.loop_length // 2)
        print(f"Load: {100 * self.stream.cpu_load:.2f}%")

    def last_sound(self, min_db, cut_db):
        # Backtrack to last onset, get everything since that onset (louder than
        # min_db) and when that sound went lower than a db cutoff (cut_db)
        pass

    def measure_air_delay(self):
        """
        Measure the air delay between speaker and microphone by playing a clave
        sound and recording it with the microphone.  Calculates the time
        difference between when the sound was played and when it was received.
        Returns the air delay in number of samples.
        """
        ll = 0 if self.anchor is None else self.anchor.loop_length
        self.actions.append(Sample(CLAVE, ll, 1.5))
        at_sample = self.rec_audio.counter
        indelay_frames = round(self.callback_time.input_delay * config.sr)
        wait_for = (
            200
            - round(self.callback_time.output_delay * 1000)
            + round(self.callback_time.input_delay * 1000)
        )
        sd.sleep(wait_for)
        after = self.rec_audio.counter
        frames_waited = after - at_sample
        rec_audio = self.rec_audio[-frames_waited + indelay_frames :]
        delay = rec_audio.sum(-1).argmax()
        return delay


class ExtraOutput:
    def __init__(self, loop: Loop):
        self.loop = loop
        self.callback_time: StreamTime = None
        self.stream = sd.OutputStream(
            samplerate=config.sr,
            device=config.headphone_device,
            channels=config.channels,
            callback=self._get_callback(),
            latency=config.latency * 0.1,
            blocksize=config.blocksize,
        )
        self.start = False
        self.stream.start()
        sd.sleep(500)
        self.sync_time = (
            self.loop.callback_time.current - self.callback_time.current
        )
        ad = loop.measure_air_delay()
        self.align(ad / config.sr)

    def _get_callback(self):
        """
        Creates callback function for this loop.
        """

        def callback(outdata, frames, time, status):
            if status:
                print(status)

            self.callback_time = StreamTime(time, 0, frames)
            if not self.start:
                outdata[:] = 0.0
            else:
                _, outdata[:] = self.loop.last_out.popleft()

        return callback

    def align(self, air_delay=config.air_delay):
        self_od = -self.callback_time.output_delay
        loop_od = -self.loop.callback_time.output_delay
        self.add_delay = loop_od + air_delay - self_od
        while True:
            ct, audio = self.loop.last_out[0]
            if (
                ct.output + self.add_delay
                < self.callback_time.output + self.sync_time
            ):
                if len(self.loop.last_out) > 1:
                    self.loop.last_out.popleft()
                else:
                    print(
                        f"Headphone device's output delay ({self_od:.4f}s) is "
                        "longer than reference output delay + air delay "
                        f"({loop_od+air_delay:.4f}s)!\nMismatch: "
                        f" {self.add_delay:.4f}s"
                    )
                    self.start = True
                    break
            else:
                self.start = True
                break


def quantize(
    sample, loop_length, start=True, lenience=config.sr * 0.2
) -> (int, int):
    """Quantize start or end recording marker to the loop boundary if
    within some interval from them.  Also returns difference between
    original sample and quantized sample.

    :param sample: start or end recording marker
    :param start: True for start, or False for end
    :param lenience: quantize if within this many seconds from the loop
        boundary

        For example, for sr=48000, the lenience (indata_at 200ms) is 9600
        samples.  If the end marker is indata_at between 38400 and 57600,
        it will instead be set to 48000, the full loop.
    """
    loop_n, frame_rem = np.divmod(sample, loop_length)
    if start:
        if sample < lenience:
            return 0, -sample
        elif sample > (loop_length - lenience):
            return 0, loop_length - sample
        else:
            return sample, 0
    else:
        if frame_rem < lenience:
            return loop_n * loop_length, -frame_rem
        elif frame_rem > (loop_length - lenience):
            return (loop_n + 1) * loop_length, loop_length - frame_rem
        else:
            return sample, 0
