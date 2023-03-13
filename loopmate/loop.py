from __future__ import annotations

import asyncio
import queue
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from scipy import signal as sig

from loopmate import config
from loopmate.actions import Actions, Start, Stop
from loopmate.utils import StreamTime

blend_windowsize = round(config.blend_length * config.sr)
RAMP = np.linspace(1, 0, blend_windowsize, dtype=np.float32)[:, None]
POP_WINDOW = sig.windows.hann(int(config.sr * config.blend_length))[:, None]

# TODO: BPM sync & metronome anchor


@dataclass
class Audio:
    # store info about different audio files to loop, perhaps with actions
    # around them as well, i.e. meter, bpm, etc.
    audio: np.ndarray = field(repr=False)
    loop_length: int | None = None
    pos_start: int = 0
    current_frame: int = 0
    remove_pop: bool = True

    def __post_init__(self):
        self.n, self.channels = self.audio.shape
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

        # Remove pop at loop edge
        if self.remove_pop:
            n_pw = len(POP_WINDOW) // 2
            self.audio[:n_pw] *= POP_WINDOW[:n_pw]
            self.audio[-n_pw:] *= POP_WINDOW[-n_pw:]

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
    def __init__(self, anchor: Audio | None = None, aioloop=None):
        self.audios = []
        self.new_audios = queue.Queue()
        self.anchor = anchor
        if anchor is not None:
            self.audios.append(anchor)

        # Always record the latest second buffers so we can look a little into
        # the past when recording
        self.recent_audio = queue.deque(
            maxlen=int(np.ceil(config.sr / config.blocksize))
        )
        self.recording = None

        # Global actions applied to fully mixed audio
        self.actions = Actions(aioloop)

        self.stream = sd.Stream(
            samplerate=config.sr,
            device=config.device,
            channels=config.channels,
            callback=self._get_callback(),
            latency=config.latency,
            blocksize=config.blocksize,
        )
        self.callback_time = None
        self.last_out = None

    def add_track(self, audio):
        if len(self.audios) == 0:
            if not isinstance(audio, Audio):
                audio = Audio(audio)
            self.anchor = audio
            self.audios.append(self.anchor)
        else:
            if not isinstance(audio, Audio):
                audio = Audio(audio, self.anchor.loop_length)
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

            # These times/frame refer to the frame that is processed in this
            # callback
            self.callback_time = StreamTime(time, current_frame)

            # Copy necessary as indata arg is passed by reference
            indata = indata.copy()
            self.recent_audio.append(indata)
            if self.recording is not None:
                self.recording.append(indata)

            outdata[:] = 0.0
            for audio in self.audios:
                a = audio.get_n(frames)
                outdata[:] += a

            if self.anchor is not None:
                self.actions.run(
                    outdata, current_frame, self.anchor.current_frame
                )

            # Align current frame for newly put audio
            # TODO: only add to new_audios when appropriate - done?
            for i in range(self.new_audios.qsize()):
                audio = self.new_audios.get()
                current_loop_iter = audio.current_frame // audio.loop_length
                audio.current_frame = (
                    current_loop_iter * audio.loop_length
                    + self.anchor.current_frame
                )

            # Store last output buffer to potentially send a slightly delayed
            # version to headphones (to match the speaker sound latency)
            self.last_out = outdata.copy()

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
                list(self.recent_audio),
                self.callback_time,
                self.stream.time,
                self.anchor.loop_length if self.anchor is not None else None,
            )
        else:
            print("REC FINISH")
            audio, remaining = self.recording.finish(t, self.callback_time)
            self.add_track(audio)
            wait_for = (remaining + config.latency) / config.sr
            if remaining > 0:
                # await asyncio.sleep(wait_for * 2)
                sd.sleep(int(wait_for * 2 * 1000))
                audio.audio[-remaining:] = np.concatenate(
                    self.recording.recordings
                )[:remaining]
            self.recording.antipop(audio)
            self.recording = None
        print(f"Load: {100 * self.stream.cpu_load:.2f}%")


class Recording:
    def __init__(
        self,
        recordings: list[np.ndarray],
        callback_time: StreamTime,
        start_time: float,
        loop_length: Audio | None = None,
    ):
        # TODO: If there are added/dropped frames during recording, may have to
        # account for them

        lengths = [len(x) for x in recordings]

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

        # This is the actual recording array index of reference_frame,
        # accounting for the delay from playing until getting input. Should
        # this perhaps just be input delay? at least when we're not recording
        # back the track - the output delay probably has to be accounted for in
        # the output, as in, playing it that many frames earlier

        indata_at = (
            sum(lengths[:-1])
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
            indata_at += move
        else:
            self.start_frame = 0
            self.loop_length = None

        self.indata_start = indata_at
        self.recordings = recordings

    def append(self, indata):
        self.recordings.append(indata)

    def finish(self, t, callback_time):
        lengths = [len(x) for x in self.recordings]

        # Between pressing and the time the last callback happened are this
        # many frames
        frames_since = round(callback_time.timediff(t) * config.sr)
        indata_at = (
            sum(lengths[:-1])
            + frames_since
            + round(callback_time.input_delay * config.sr)
        )
        n = indata_at - self.indata_start
        if self.loop_length is not None:
            reference_frame = self.start_frame + n
            self.end_frame, move = self.quantize(reference_frame, False)
            print(f"\n\rMove {move} to {self.end_frame}")
            indata_at += move
        else:
            self.end_frame = self.loop_length = n

        recording = np.zeros(
            (indata_at - self.indata_start, self.recordings[0].shape[1]),
            dtype=np.float32,
        )
        recordings = np.concatenate(self.recordings)
        self.recordings.clear()
        # To potentially do a more graceful blending at loop boundaries
        self.end_xfade = recordings[
            self.indata_start - config.blend_frames : self.indata_start
        ]
        if len(self.end_xfade) < config.blend_frames:
            self.end_xfade = np.zeros(
                (config.blend_frames, recording.shape[1]), np.float32
            )

        # This is expressed in the full dimensions of recordings
        available = min(len(recordings), indata_at)
        recording[: available - self.indata_start] = recordings[
            self.indata_start : available
        ]
        remaining = indata_at - available

        audio = Audio(
            recording,
            self.loop_length,
            self.start_frame,
            0,
            remove_pop=False,
        )
        # We need to add the starting frame (in case we start this audio late)
        # as well as subtract the audio delay we added when we started
        # recording
        n += audio.pos_start - round(callback_time.output_delay * config.sr)
        if n > audio.n_loop_iter * self.loop_length:
            n = n % self.loop_length
        audio.current_frame = n
        print(audio)

        return audio, remaining

    def antipop(self, audio):
        # TODO: always crossfade, depends on whether full or partial loop
        if (self.end_frame % self.loop_length) == 0:
            audio.audio[-config.blend_frames :] = (
                RAMP * audio.audio[-config.blend_frames :]
                + (1 - RAMP) * self.end_xfade
            )
        else:
            n_pw = len(POP_WINDOW) // 2
            audio.audio[:n_pw] *= POP_WINDOW[:n_pw]
            audio.audio[-n_pw:] *= POP_WINDOW[-n_pw:]

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
