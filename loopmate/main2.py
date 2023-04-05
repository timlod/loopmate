from __future__ import annotations

import threading
import time
from multiprocessing import Condition, Event, Process

import mido
import numpy as np
import pedalboard
import sounddevice as sd
import soundfile as sf

from loopmate import circular_array, config, utils
from loopmate.actions import (
    BackCaptureTrigger,
    Effect,
    Mute,
    MuteTrigger,
    RecordTrigger,
)
from loopmate.loop import Audio, ExtraOutput, Loop

delay = pedalboard.Delay(0.8, 0.4, 0.3)


class MidiQueue:
    def __init__(self, loop):
        self.loop = loop
        self.port = mido.open_input(callback=self.receive)

    def receive(self, message):
        gain = 1.0
        if message.type == "note_on":
            t = time.time()
            print(time.time() - t)
            if message.note == 25:
                self.loop.start()
            elif message.note == 35:
                if len(self.loop.actions.actions) > 0 and isinstance(
                    self.loop.actions.actions[0], Mute
                ):
                    self.loop.actions.actions[0].cancel()
                else:
                    self.loop.actions.actions.appendleft(
                        Mute(
                            self.loop.anchor.current_frame, self.loop.anchor.n
                        )
                    )
            elif message.note == 27:
                self.loop.audios.pop()
                if len(self.loop.audios) == 0:
                    self.loop.anchor = None
            elif message.note == 47:
                self.loop.record()
            elif message.note == 57:
                n = self.loop.anchor.loop_length
                when = (
                    n - round(self.loop.callback_time.output_delay * config.sr)
                ) % self.loop.anchor.loop_length
                self.loop.actions.actions.append(
                    RecordTrigger(when, n, spawn=RecordTrigger(when, n))
                )
                self.loop.actions.actions.append(
                    MuteTrigger(when, n, spawn=MuteTrigger(when, n))
                )
                print(self.loop.actions.actions)
            elif message.note == 30:
                n = self.loop.anchor.loop_length
                when = n - config.blend_frames - 256
                self.loop.actions.actions.append(
                    MuteTrigger(when, n, loop=False)
                )
            elif message.note == 48:
                self.loop.backcapture(1)
            elif message.note == 43:
                self.loop.measure_air_delay()
            elif message.note == 50:
                self.loop.audios[-1].audio *= 1.2
            elif message.note == 52:
                self.loop.audios[-1].audio *= 0.8
            elif message.note == 55:
                self.loop.audios[-1].reset_audio()
            elif message.note == 53:
                self.loop.audios[-1].audio = delay(
                    self.loop.audios[-1].audio, config.sr, reset=False
                )


def plan_callback(loop):
    while True:
        trigger = loop.actions.plans.get()
        if isinstance(trigger, RecordTrigger):
            print("Record in plan_callback")
            loop.record()
            continue
        elif isinstance(trigger, BackCaptureTrigger):
            loop.backcapture(trigger.n_loops)
            continue


def analyze(arr, stop_event, cond):
    try:
        with cond:
            while not stop_event.is_set():
                cond.wait()
                arr.fft()
    except Exception as e:
        print("stopped sharing")
        arr.stop_sharing()
        raise e


if __name__ == "__main__":
    sa = circular_array.CircularArraySTFT(
        config.sr * config.max_recording_length, config.channels
    )
    sa.make_shared(create=True)
    se = Event()
    cond = Condition()
    ap = Process(target=analyze, args=(sa, se, cond))
    ap.start()
    print("started")
    print(sa)

    print(sd.query_devices())
    piano, _ = sf.read("../data/piano.wav", dtype=np.float32)
    clave, _ = sf.read("../data/clave.wav", dtype=np.float32)
    clave = np.concatenate(
        (
            1 * clave[:, None],
            np.zeros((config.sr - len(clave), 1), dtype=np.float32),
        )
    )
    loop = Loop(Audio(clave), cond)
    loop.start()
    # hl = ExtraOutput(loop)

    print(loop)

    ps = pedalboard.PitchShift(semitones=-6)
    ds = pedalboard.Distortion(drive_db=20)
    delay = pedalboard.Delay(0.8, 0.1, 0.3)
    limiter = pedalboard.Limiter()
    loop.actions.append(Effect(0, 10000000, lambda x: limiter(x, config.sr)))

    midi = MidiQueue(loop)

    plan_thread = threading.Thread(target=plan_callback, args=(loop,))
    plan_thread.start()
