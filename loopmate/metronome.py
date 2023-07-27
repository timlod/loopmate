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
CLAVE = CLAVE[:2000, None]
CLAVE[-256:, 0] *= np.linspace(1, 0, 256)

# Shorten the sample somewhat and raise pitch
CLAVE = resample(CLAVE, 1800)
CLAVE1 = resample(CLAVE, 1500)


class Metronome(loop.Audio):
    """Heavily modified loop Audio to allow for programmed metronome patterns.
    Clicks are placed by samples triggered at click times.
    """

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
        self.schedule = None
        self.tempo_schedule = None

        # 4096 would be the max buffer size to be safe
        audio = np.zeros(4096, dtype=np.float32)
        super().__init__(audio)
        self.n = self.loop_length = self.length()
        self.add_clicks()

    def add_schedule(self, schedule):
        self.schedule = schedule
        self.actions.prepend(self.schedule)

    def add_tempo_schedule(self, tempo_schedule):
        self.tempo_schedule = tempo_schedule
        self.actions.append(self.tempo_schedule)

    def generate_click_locations(self):
        """Generate the locations of clicks.

        Permutation 0 will always return quarter note locations.  For each
        subsequent subdivision (i.e. subdivision 2 corresponds to eighth notes), we
        can get subdivision - 1 permutations, i.e. permutation 1 at subdivision 2
        will correspond to eighth note off-beats.

        At subdivision 4 (sixteenth notes), permutation 3 will be the last
        sixteenth at each beat.
        """
        if isinstance(self.permutation, int):
            permutation = [self.permutation]
        else:
            permutation = self.permutation

        for p in permutation:
            assert (
                p < self.subdivision
            ), "permutation needs to be < subdivision!"

        samples_per_beat = (60 / self.bpm) * self.sr

        clicks = []
        for p in permutation:
            clicks.extend(
                [
                    round(samples_per_beat * i / self.subdivision)
                    for i in range(
                        p, self.beats * self.subdivision, self.subdivision
                    )
                ]
            )
        return clicks

    def length(self):
        return round(self.beats * 60 / self.bpm * self.sr)

    @property
    def tempo(self):
        return self.bpm

    @tempo.setter
    def tempo(self, bpm: int):
        """Set tempo to given BPM.

        If a schedule is running that sets BPM, this will be overwritten.  If
        the schedule doesn't contain any BPM, it will apply to the entire
        schedule.

        :param bpm: BPM
        """
        assert bpm > 0, "bpm can't be smaller than 1!"
        old_length = self.loop_length
        self.bpm = bpm
        self.loop_length = self.n = self.length()
        # Translate the current index onto the new timescale
        ci_frac = self.current_index / old_length
        self.current_index = (
            round(self.loop_length * ci_frac) if ci_frac else 0
        )
        self.add_clicks()

    def set(self, d):
        """Set all parameters given a dictionary.  Used mainly for schedules
        which are updated 'on the one', hence doesn't use index translation
        like in the tempo setter.

        :param d: dictionary containing some of
                  beats/subdivision/permutation/bpm/p
        """
        self.beats = d["beats"] if "beats" in d else self.beats
        self.subdivision = (
            d["subdivision"] if "subdivision" in d else self.subdivision
        )
        self.permutation = (
            d["permutation"] if "permutation" in d else self.permutation
        )
        self.p = d["p"] if "p" in d else self.p
        self.bpm = d["bpm"] if "bpm" in d else self.bpm
        self.loop_length = self.n = self.length()
        self.add_clicks()

    def add_clicks(self):
        self.actions.actions.clear()
        # Re-add schedule triggers
        if self.schedule is not None:
            self.actions.append(self.schedule)
        if self.tempo_schedule is not None:
            self.actions.append(self.tempo_schedule)

        clicks = self.generate_click_locations()

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

        # Here we place clicks through through actions
        self.actions.run(out, current_index, self.current_index)
        return out


class ClickTrigger(Trigger):
    def __init__(self, when, loop_length, p=1.0, **kwargs):
        """Trigger to play a click sample at an exact point in time.

        :param when: index at which to play the sample
        :param loop_length: length of the containing loop
        :param p: probability with which to trigger - use to randomly drop out
                  clicks.
        """
        self.p = p
        super().__init__(when, loop_length, loop=True, **kwargs)

    def do(self, actions, current_index):
        # TODO: potential logic to always play/leave out the one in rand
        if random.random() <= self.p:
            sample = Sample(
                CLAVE if self.when != 0 else CLAVE1,
                # In case of drastic slowdown this prevents premature stopping
                self.loop_length * 10,
                wait=max(0, self.when - current_index),
            )
            actions.actions.appendleft(sample)
            actions.active.put_nowait(sample)


class ClickSchedule(Trigger):
    def __init__(self, metronome: Metronome, schedule: list[dict], **kwargs):
        """Schedule containing complex click patterns.

        :param metronome: Metronome instance
        :param schedule: list of dictionaries containing the click pattern
        """
        self.metronome = metronome
        schedule = [x for x in schedule for i in range(x["bars"])]
        self.schedule = Queue()
        for x in schedule:
            self.schedule.put(x)

        self.metronome.current_index = 0
        self.metronome.set(schedule[0])
        super().__init__(
            0, self.metronome.loop_length, loop=True, priority=0, **kwargs
        )

    def do(self, actions, current_index):
        x = self.schedule.get()
        while not self.metronome.actions.active.empty():
            self.metronome.actions.active.get()

        self.metronome.set(x)
        self.schedule.put(x)
        clicks = self.metronome.generate_click_locations()

        self.loop_length = self.metronome.loop_length
        # In case we need to immediately play on the one
        if clicks[0] == 0:
            # Schedules come before clicks, skip those
            for action in self.metronome.actions.actions:
                if isinstance(action, ClickTrigger):
                    self.metronome.actions.active.put_nowait(action)
                    return


class TempoSchedule(Trigger):
    def __init__(
        self,
        metronome: Metronome,
        min_bpm: int,
        max_bpm: int,
        step: int,
        bars_per_step: int,
        mode: str = "repeat",
        **kwargs,
    ):
        """Schedule to raise or decrease tempo in a controlled manner.

        :param metronome: Metronome instance
        """
        self.metronome = metronome
        self.times = np.repeat(range(min_bpm, max_bpm, step), bars_per_step)
        if mode == "repeat":
            self.tempo = repeat_generator(self.times)
        else:
            self.tempo = pingpong_generator(self.times)

        # Tempo schedule needs to apply before Click schedule
        super().__init__(
            0, self.metronome.loop_length, loop=True, priority=-1, **kwargs
        )

    def do(self, actions, current_index):
        print(self.metronome.tempo)
        self.metronome.tempo = next(self.tempo)
        self.loop_length = self.metronome.loop_length


def repeat_generator(x):
    while True:
        for i in x:
            yield i


def pingpong_generator(x):
    while True:
        for i in x:
            yield i
        for i in reversed(x):
            yield i


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
        m.add_schedule(m_sched)
        m.add_tempo_schedule(TempoSchedule(m, 120, 240, 10, 1, "reverse"))
        loop.start()
        midi = MIDIHandler(m, loop)
        try:
            while True:
                sd.sleep(10)
        except KeyboardInterrupt:
            pass
