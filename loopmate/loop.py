from __future__ import annotations

import queue
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from scipy import signal as sig

from loopmate import config
from loopmate.actions import Actions, Sample, Start, Stop
from loopmate.circular_array import CircularArray
from loopmate.utils import CLAVE, StreamTime

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
    current_frame: int = 0
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
                2 * np.ceil(np.log2(self.n / self.loop_length))
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

    def get_n(self, frames: int):
        """Return the next batch of audio in the loop

        :param frames: number of audio samples to return
        """
        leftover = self.n - self.current_frame
        chunksize = min(leftover, frames)
        current_frame = self.current_frame

        if leftover <= frames:
            out = self.audio[
                np.r_[
                    current_frame : current_frame + chunksize,
                    : frames - leftover,
                ]
            ]
            self.current_frame = frames - leftover
        else:
            out = self.audio[current_frame : current_frame + chunksize]
            self.current_frame += frames

        self.actions.run(out, current_frame, self.current_frame)
        return out

    def reset_audio(self):
        self.audio = self._audio.copy()

    def make_anchor(self):
        self.loop_length = self.loop_length * self.n_loop_iter
        self.n_loop_iter = 1


class Loop:
    def __init__(self, anchor: Audio | None = None, mp_cond=None):
        self.mp_cond = mp_cond
        self.audios = []
        self.new_audios = queue.Queue()
        self.anchor = anchor
        if anchor is not None:
            self.audios.append(anchor)

        # Always record audio buffers so we can easily look back for loopables
        self.recent_audio = CircularArray(
            config.sr * config.max_recording_length, config.channels
        )
        self.recent_audio.make_shared()

        self.recording = None

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
            audio.actions.append(Start(audio.current_frame, audio.loop_length))
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
                current_frame = 0
            else:
                current_frame = self.anchor.current_frame

            # Align current frame for newly put audio - maybe not necessary now
            for i in range(self.new_audios.qsize()):
                audio = self.new_audios.get()
                current_loop_iter = audio.current_frame // audio.loop_length
                audio.current_frame = (
                    current_loop_iter * audio.loop_length + current_frame
                )

            # These times/frame refer to the frame that is processed in this
            # callback
            self.callback_time = StreamTime(time, current_frame, frames)

            # Copy necessary as indata arg is passed by reference
            indata = indata.copy()
            self.recent_audio.write(indata)

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
                0 if self.anchor is None else self.anchor.current_frame
            )
            self.actions.run(outdata, current_frame, next_frame)
            # Notify whenever we finished one iteration
            with self.mp_cond:
                self.mp_cond.notify()

        return callback

    def start(self):
        self.stream.stop()
        # self.current_frame = 0
        if self.anchor is not None:
            self.actions.actions.append(
                Start(self.anchor.current_frame, self.anchor.loop_length)
            )
        self.stream.start()

    def stop(self):
        if self.anchor is not None:
            print(f"Stopping stream action. {self.anchor.current_frame}")
            self.actions.actions.append(Stop(self.anchor.current_frame))

    def record(self):
        t = self.stream.time
        if self.recording is None:
            print("REC START")
            self.recording = Recording(
                self.recent_audio,
                self.callback_time,
                self.stream.time,
                self.anchor.loop_length if self.anchor is not None else None,
            )
        else:
            # If no loop, use bpm quantization
            print("REC FINISH")
            audio = self.recording.finish(t, self.callback_time)
            if audio is None:
                print("Empty recording!")
            else:
                self.add_track(audio)
            self.recording = None
        print(f"Load: {100 * self.stream.cpu_load:.2f}%")

    def backcapture(self, n):
        print(f"Backcapture {n=}!")
        t = self.stream.time
        recording = Recording(
            self.recent_audio,
            self.callback_time,
            self.stream.time,
            self.anchor.loop_length if self.anchor is not None else None,
            # Choosing lenience like this makes sure that if we're before
            # halfway through the next loop, we still use the previous one -
            # otherwise, we record the currently ongoing loop
            lenience=self.anchor.loop_length // 2,
        )
        recording.rec_start -= recording.loop_length * n
        audio = recording.finish(t, self.callback_time)
        self.add_track(audio)
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
        at_sample = self.recent_audio.counter
        indelay_frames = round(self.callback_time.input_delay * config.sr)
        wait_for = (
            200
            - round(self.callback_time.output_delay * 1000)
            + round(self.callback_time.input_delay * 1000)
        )
        sd.sleep(wait_for)
        after = self.recent_audio.counter
        frames_waited = after - at_sample
        recent_audio = self.recent_audio[-frames_waited + indelay_frames :]
        delay = recent_audio.sum(-1).argmax()
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


class Recording:
    def __init__(
        self,
        rec: CircularArray,
        callback_time: StreamTime,
        start_time: float,
        loop_length: int | None = None,
        lenience: int = round(config.sr * 0.2),
    ):
        self.lenience = lenience
        # Between pressing and the time the last callback happened are this
        # many frames
        self.rec_start, frames_since = self.recording_event(
            callback_time, start_time
        )

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

        # Quantize to loop_length if it exists
        if loop_length is not None:
            self.loop_length = loop_length
            # Wrap around if we shifted beyond loop_length
            if reference_frame > loop_length:
                reference_frame -= loop_length
            self.start_frame, move = quantize(
                reference_frame, loop_length, lenience=lenience
            )
            print(
                f"\n\rMoving {move} from {reference_frame} to {self.start_frame}"
            )
            self.rec_start += move
        else:
            self.start_frame = 0
            self.loop_length = None

        self.rec = rec

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

    def finish(self, t, callback_time):
        # Between pressing and the time the last callback happened are this
        # many frames
        self.rec_stop, _ = self.recording_event(callback_time, t)

        n = self.rec_stop - self.rec_start
        if self.loop_length is not None:
            reference_frame = self.start_frame + n
            self.end_frame, move = quantize(
                reference_frame, self.loop_length, False, self.lenience
            )
            print(f"\n\rMove {move} to {self.end_frame}")
            self.rec_stop += move
            n += move
        else:
            self.end_frame = self.loop_length = n

        if n == 0:
            return None
        if self.rec_stop > self.rec.counter:
            wait_for = (
                self.rec_stop - self.rec.counter + 3 * config.blocksize
            ) / config.sr
            print(f"Missing {self.rec_stop - self.rec.counter} frames.")
            print(f"Waiting {wait_for}s.")
            # Need to specify sleep time in ms, add blocksize to make sure we
            # don't get a block too few
            sd.sleep(int(wait_for * 1000))
            print(f"c: {self.rec.counter}, s: {self.rec_stop}")
            assert self.rec.counter >= self.rec_stop

        back = self.rec.frames_since(self.rec_stop)
        rec_i = -(n + back)
        print(
            f"{rec_i=}, {back=}, {n=}, {self.rec_stop=}, {self.rec.counter=}"
        )
        recording = self.rec[rec_i:-back]
        self.antipop(recording, self.rec[rec_i - config.blend_frames : rec_i])

        # We need to add the starting frame (in case we start this audio late)
        # as well as subtract the audio delay we added when we started
        # recording
        n_loop_iter = int(2 * np.ceil(np.log2(n / self.loop_length)))
        n += self.start_frame - round(callback_time.output_delay * config.sr)
        if n > n_loop_iter * self.loop_length:
            n = n % self.loop_length
        audio = Audio(recording, self.loop_length, current_frame=n)
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


def quantize(
    frame, loop_length, start=True, lenience=config.sr * 0.2
) -> (int, int):
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
    loop_n, frame_rem = np.divmod(frame, loop_length)
    if start:
        if frame < lenience:
            return 0, -frame
        elif frame > (loop_length - lenience):
            return 0, loop_length - frame
        else:
            return frame, 0
    else:
        if frame_rem < lenience:
            return loop_n * loop_length, -frame_rem
        elif frame_rem > (loop_length - lenience):
            return (loop_n + 1) * loop_length, loop_length - frame_rem
        else:
            return frame, 0
