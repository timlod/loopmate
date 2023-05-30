from __future__ import annotations

import threading
import time
from multiprocessing import Condition, Event, Process

import mido
import numpy as np
import pedalboard
import sounddevice as sd
import soundfile as sf

from loopmate import circular_array, config, recording as lr, utils
from loopmate.actions import (
    BackCaptureTrigger,
    Effect,
    Mute,
    MuteTrigger,
    RecordTrigger,
)
from loopmate.loop import Audio, ExtraOutput, Loop

mido.ports.set_sleep_time(0)
delay = pedalboard.Delay(0.8, 0.4, 0.3)


class MidiQueue:
    def __init__(self, loop):
        self.loop = loop
        self.port = mido.open_input(callback=self.receive)
        self.in_rec = False

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
                if self.loop.anchor is None:
                    bpm_quant = True
                else:
                    bpm_quant = False
                self.in_rec = not self.in_rec
                # Record should probably take start/end frames optionally as
                # input which could come from bpm quantization
                if self.in_rec:
                    self.loop.start_record()
                else:
                    self.loop.stop_record()
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
            elif message.note == 41:
                self.loop.recording.data.quit = True
                self.loop.actions.plans.put_nowait(True)
                # Need to clean up all shared memory :/
                del self.loop.rec_audio
                self.loop.stop()


def plan_callback(loop: Loop):
    while True:
        trigger = loop.actions.plans.get()
        if isinstance(trigger, RecordTrigger):
            print("Record in plan_callback")
            loop.record()
            continue
        elif isinstance(trigger, BackCaptureTrigger):
            loop.backcapture(trigger.n_loops)
            continue
        elif isinstance(trigger, bool):
            break


def analysis():
    with lr.RecAnalysis(config.rec_n, config.channels) as rec:
        rec.run()
    print("done analysis")


def a2():
    with lr.RecA(config.rec_n, config.channels) as rec:
        rec.run()
    print("done a2")


if __name__ == "__main__":
    with lr.RecMain(config.rec_n, config.channels) as rec:
        ap = Process(target=analysis)
        ap2 = Process(target=a2)
        ap.start()
        ap2.start()

        print("started")

        print(sd.query_devices())
        piano, _ = sf.read("../data/piano.wav", dtype=np.float32)
        clave, _ = sf.read("../data/clave.wav", dtype=np.float32)
        clave = np.concatenate(
            (
                1 * clave[:, None],
                np.zeros((config.sr - len(clave), 1), dtype=np.float32),
            )
        )
        loop = Loop(rec, Audio(clave))
        loop.start()
        # hl = ExtraOutput(loop)

        print(loop)

        ps = pedalboard.PitchShift(semitones=-6)
        ds = pedalboard.Distortion(drive_db=20)
        delay = pedalboard.Delay(0.8, 0.1, 0.3)
        limiter = pedalboard.Limiter()
        loop.actions.append(
            Effect(0, 10000000, lambda x: limiter(x, config.sr))
        )

        midi = MidiQueue(loop)

        plan_thread = threading.Thread(target=plan_callback, args=(loop,))
        plan_thread.start()
        ap.join()
        ap2.join()
        plan_thread.join()
        sd.sleep(10)
