import random
import soundfile as sf
from warnings import warn
import numpy as np
import sounddevice as sd
from loopmate import loop
from loopmate.actions import Sample, Trigger
import rtmidi


CLAVE, MSR = sf.read("../data/clave.wav", dtype=np.float32)
CLAVE = CLAVE[:, None]


class Metronome(loop.Audio):
    # this will be the anchor beat, with samples fired as repeated actions
    def __init__(
        self,
        bpm: int,
        beats: int,
        subdivision: int,
        sr: int = 44100,
    ):
        self.bpm = bpm
        self.beats = beats
        self.subdivision = subdivision
        self.sr = sr

        # 4096 would be the max buffer size to be safe
        audio = np.zeros(4096, dtype=np.float32)
        super().__init__(audio)
        self.n = self.loop_length = self.length()
        self.new_actions()

    def length(self):
        return round(self.beats * 60 / self.bpm * self.sr)

    @property
    def tempo(self):
        return self.bpm

    @tempo.setter
    def tempo(self, bpm):
        assert bpm > 0, "bpm can't be smaller than 1!"
        old_length = self.loop_length
        self.bpm = bpm
        self.loop_length = self.n = self.length()
        # need to translate the current index onto the new timescale!
        ci_frac = self.current_index / old_length
        self.current_index = (
            round(self.loop_length * ci_frac) if ci_frac else 0
        )
        self.new_actions()

    def new_actions(self):
        # whenever changes are made, this should purge the action list and make
        # a new one according to the metronome settings
        self.actions.actions.clear()
        for beat in range(self.beats):
            self.actions.append(
                ClickTrigger(
                    round(beat * 60 / self.bpm * self.sr), self.loop_length
                )
            )

    def get_n(self, samples: int) -> np.ndarray:
        """Return the next batch of audio in the loop

        :param samples: number of audio samples to return
        """
        leftover = self.loop_length - self.current_index
        current_index = self.current_index

        if leftover <= samples:
            self.current_index = samples - leftover
        else:
            self.current_index += samples

        # self.audio is just zeros, so this will always work
        out = self.audio[:samples].copy()

        # this will actually place our clicks
        self.actions.run(out, current_index, self.current_index)
        return out


class ClickTrigger(Trigger):
    def __init__(self, when, loop_length, p=1.0, **kwargs):
        """Trigger to immediately play a click sample.

        :param when: index at which to play the sample
        :param loop_length: length of the containing loop
        :param p: probability with which to trigger - use to randomly drop out
                  clicks.
        """
        self.p = p
        super().__init__(when, loop_length, loop=True, **kwargs)

    def do(self, actions):
        # potentially just blow up this loop_length in case we drastically slow
        # down
        if random.random() <= self.p:
            sample = Sample(CLAVE, self.loop_length * 10, 0.9)
            actions.actions.appendleft(sample)
            actions.active.put_nowait(sample)


def decode_midi_status(status: int) -> (int, int):
    """
    Convert int64 containing the MIDI command and channel into those
    respectively.

    :param status: MIDI status as returned by python-rtmidi
    """
    return status // 16, status % 16 + 1


class MIDIHandler:
    """
    Defines callback to attach to MIDI input as well as possible
    commands/actions to perform given MIDI input.

    TODO: Make this prettier, logical - currently it's based on a default VMPK
    map.  Should read a config with midi mapping to allow easy use with
    different MIDI devices.
    """

    def __init__(self, metronome: Metronome, loop):
        self.loop = loop
        self.metronome = metronome
        self.port = rtmidi.MidiIn().open_port(config.MIDI_PORT)
        self.port.set_callback(self.receive)
        self.in_rec = False

    def receive(self, event: tuple[list[int, int, int], float], data=None):
        """Callback used by rtmidi.

        :param event: event containing [status, note, velocity], deltatime.
            Can have different structure if special MIDI events are fired.
            Those are currently ignored.
        :param data: additional data
        """
        try:
            [status, note, velocity], deltatime = event
            command, channel = decode_midi_status(status)
        except Exception as e:
            warn(f"{event} was not decodable!\n{e.message}")

        if command != 9:
            return

        if config.MIDI_CHANNEL > 0:
            if channel != config.MIDI_CHANNEL:
                return
        self.command(note)

    def command(self, note: int):
        """Interact with self.loop based on the MIDI noteon event received.

        TODO: add different actions based on MIDI velocity for live performance

        :param note: MIDI note
        """
        match note:
            case 25:
                self.metronome.tempo = self.metronome.tempo + 1
                print(self.metronome.tempo)
            case 27:
                self.metronome.tempo = self.metronome.tempo - 1
                print(self.metronome.tempo)
            case 47:
                self.metronome.tempo = self.metronome.tempo + 10
                print(self.metronome.tempo)
            case 57:
                self.metronome.tempo = self.metronome.tempo - 10
                print(self.metronome.tempo)
            case 41:
                self.loop.rec.data.quit = True
                self.loop.actions.plans.put_nowait(True)
                del self.loop.rec_audio
                self.loop.stop()
            case _:
                pass


if __name__ == "__main__":
    from loopmate import config, recording as lr

    with lr.RecAudio(config.REC_N, config.N_CHANNELS) as rec:
        m = Metronome(60, 1, 4, config.SR)
        loop = loop.Loop(rec, m)
        loop.start()
        midi = MIDIHandler(m, loop)
        while True:
            sd.sleep(10)
