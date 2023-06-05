from __future__ import annotations

import threading
import time
from multiprocessing import Process
from warnings import warn

import numpy as np
import pedalboard
import rtmidi
import sounddevice as sd
import soundfile as sf

from loopmate import config, recording as lr
from loopmate.actions import (
    BackCaptureTrigger,
    Effect,
    Mute,
    MuteTrigger,
    RecordTrigger,
)
from loopmate.loop import Audio, Loop

delay = pedalboard.Delay(0.8, 0.4, 0.3)


def decode_midi_status(status):
    return status // 16, status % 16 + 1


class MidiQueue:
    def __init__(self, loop: Loop):
        self.loop = loop
        self.port = rtmidi.MidiIn().open_port(config.midi_port)
        self.port.set_callback(self.receive)
        self.in_rec = False

    def receive(self, event, data=None):
        gain = 1.0
        try:
            [status, note, velocity], deltatime = event
            command, channel = decode_midi_status(status)
        except Exception as e:
            warn(f"{event} was not decodable!\n{e.message}")

        if command != 9:
            return

        if config.midi_channel > 0:
            if channel != config.midi_channel:
                return

        match note:
            case 25:
                self.loop.start()
            case 35:
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
            case 27:
                self.loop.audios.pop()
                if len(self.loop.audios) == 0:
                    self.loop.anchor = None
            case 47:
                if self.loop.anchor is None:
                    bpm_quant = True
                else:
                    bpm_quant = False
                self.in_rec = not self.in_rec
                if self.in_rec:
                    self.loop.start_recording(config.record_channels)
                else:
                    self.loop.stop_recording()
            case 57:
                # Trigger record action at next loop which spawns another
                # record (recording stop)
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
            case 30:
                n = self.loop.anchor.loop_length
                when = n - config.blend_samples - 256
                self.loop.actions.actions.append(
                    MuteTrigger(when, n, loop=False)
                )
            case 48:
                self.loop.backcapture(1)
            case 43:
                self.loop.measure_air_delay()
            case 50:
                self.loop.audios[-1].audio *= 1.2
            case 52:
                self.loop.audios[-1].audio *= 0.8
            case 55:
                self.loop.audios[-1].reset_audio()
            case 53:
                self.loop.audios[-1].audio = delay(
                    self.loop.audios[-1].audio, config.sr, reset=False
                )
            case 41:
                self.loop.rec.data.quit = True
                self.loop.actions.plans.put_nowait(True)
                del self.loop.rec_audio
                self.loop.stop()
            case _:
                pass  # Do nothing for other values.

        def start_new_recording(self, channels):
            self.loop.start_record(channels, new=self.loop.anchor is None)

        def delete(self, i):
            assert i < len(
                self.loop.audios
            ), f"i ({i}) is larger than the number of audios!"
            self.loop.audios.pop(i)
            if len(self.loop.audios) == 0:
                self.loop.anchor = None


def plan_callback(loop: Loop):
    while True:
        print("plan")
        trigger = loop.actions.plans.get()
        if isinstance(trigger, RecordTrigger):
            print("Record in plan_callback")
            # TODO: this will run into trouble if result_type == 8
            if loop.rec.data.result_type == 0:
                loop.start_recording()
            else:
                loop.stop_recording()
            continue
        elif isinstance(trigger, BackCaptureTrigger):
            loop.backcapture(trigger.n_loops)
            continue
        elif isinstance(trigger, bool):
            break


def analysis_target():
    with lr.RecAnalysis(config.rec_n, config.channels) as rec:
        rec.run()
    print("done analysis")


def ondemand_target():
    with lr.AnalysisOnDemand(config.rec_n, config.channels) as rec:
        rec.run()
    print("done a2")


if __name__ == "__main__":
    with lr.RecAudio(config.rec_n, config.channels) as rec:
        ap = Process(target=analysis_target)
        ap2 = Process(target=ondemand_target)
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
