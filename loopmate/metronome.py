import json
from queue import Queue
import random
import soundfile as sf
from warnings import warn
import numpy as np
import sounddevice as sd
from loopmate import loop
from loopmate.actions import Sample, Trigger
import rtmidi
from scipy.signal import resample

CLAVE, MSR = sf.read("../data/clave.wav", dtype=np.float32)
CLAVE = CLAVE[:, None]
# Shorten the sample somewhat
CLAVE = resample(CLAVE, 4096)


def generate_click_locations(
    beats: int, bpm: int, level: int = 1, permutation: int = 0, sr: int = 44100
):
    """Generate the locations of clicks.

    Permutation 0 will always return quarter note locations.  For each
    subsequent level (i.e. level 2 corresponds to eighth notes), we can get
    level - 1 permutations, i.e. permutation 1 at level 2 will correspond to
    eighth note off-beats.

    At level 4 (sixteenth notes), permutation 3 will be the last sixteenth at
    each beat.

    :param beats: number of beats (subdivision is assumed to be quarter = /4)
        in the time signature
    :param bpm: beats per minute
    :param sr: sampling rate for playback
    :param level: how many subdivisions we put into one quarter (of each beat),
        e.g. 1 - quarters, 2 - eighth, 3 - eighth triplets, 4 - sixteenth, 5 -
        quintuplets, etc.

    :param permutation: which permutation to use - default (0) always uses the
        downbeat
    """
    assert permutation < level, "permutation needs to be < level!"
    samples_per_beat = (60 / bpm) * sr

    return [
        round(samples_per_beat * i / level)
        for i in range(permutation, beats * level, level)
    ]


class Metronome(loop.Audio):
    # this will be the anchor beat, with samples fired as repeated actions
    def __init__(
        self,
        bpm: int,
        beats: int,
        subdivision: int,
        permutation: int = 0,
        sr: int = 44100,
    ):
        self.bpm = bpm
        self.beats = beats
        self.subdivision = subdivision
        self.permutation = permutation
        self.sr = sr
        self.p = 1

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

    def set(self, bpm, beats, subdivision, permutation, p):
        self.beats = beats
        self.subdivision = subdivision
        self.permutation = permutation
        self.p = p
        self.tempo = bpm

    def new_actions(self):
        # whenever changes are made, this should purge the action list and make
        # a new one according to the metronome settings
        self.actions.actions.clear()
        clicks = generate_click_locations(
            self.beats, self.bpm, self.subdivision, self.permutation, self.sr
        )

        for click in clicks:
            self.actions.append(
                ClickTrigger(click, self.loop_length, p=self.p)
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

        # self.audio is just zeros, so this will always work we. we use copy
        # because actions can modify out in place, which without copy would be
        # just a view
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

    def do(self, actions, current_index):
        # potentially just blow up this loop_length in case we drastically slow
        # down
        if random.random() <= self.p:
            sample = Sample(
                CLAVE,
                self.loop_length * 10,
                wait=self.when - current_index,
                gain=0.9,
            )
            actions.actions.appendleft(sample)
            actions.active.put_nowait(sample)


class ClickSchedule(Trigger):
    def __init__(self, metronome: Metronome, schedule, **kwargs):
        """Use the supplied click schedule

        :param when:
        :param loop_length:
        :param schedule:
        """
        self.metronome = metronome
        schedule = [x for x in schedule for i in range(x["bars"])]
        self.schedule = Queue()
        for x in schedule:
            if "bars" in x:
                del x["bars"]
            self.schedule.put(x)

        self.metronome.set(**schedule[0])
        super().__init__(0, self.metronome.loop_length, loop=True, **kwargs)

    def do(self, actions):
        x = self.schedule.get()
        self.metronome.set(**x)
        self.schedule.put(x)
        self.when = self.loop_length - 1024
        self.loop_length = self.metronome.loop_length


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

    with open("test.json") as f:
        schedule = json.load(f)

    with lr.RecAudio(config.REC_N, config.N_CHANNELS) as rec:
        m = Metronome(60, 4, 4, 3, sr=config.SR)
        loop = loop.Loop(rec, m)
        m_sched = ClickSchedule(m, schedule)
        loop.actions.append(m_sched)
        loop.start()
        midi = MIDIHandler(m, loop)
        while True:
            sd.sleep(10)
