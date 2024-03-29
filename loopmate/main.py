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
from loopmate.loop import Audio, ExtraOutput, Loop


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

    def __init__(self, loop: Loop):
        self.loop = loop
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
                    self.loop.start_recording(channels=config.RECORD_CHANNELS)
                else:
                    self.loop.stop_recording()
            case 57:
                # Trigger record action at next loop which spawns another
                # record (recording stop)
                n = self.loop.anchor.loop_length
                when = (
                    n - round(self.loop.callback_time.output_delay * config.SR)
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
                when = n - config.BLEND_SAMPLES - 256
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
                    self.loop.audios[-1].audio, config.SR, reset=False
                )
            case 41:
                self.loop.rec.data.quit = True
                self.loop.actions.plans.put_nowait(True)
                del self.loop.rec_audio
                self.loop.stop()
            case _:
                pass

        def delete(self, i):
            assert i < len(
                self.loop.audios
            ), f"i ({i}) is larger than the number of audios!"
            self.loop.audios.pop(i)
            if len(self.loop.audios) == 0:
                self.loop.anchor = None


def plan_callback(loop: Loop):
    """Callback which picks up triggers/actions from the plan queue.

    :param loop: Loop object containing Actions
    """
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
    """
    target function for the multiprocessing.Process which will run ongoing
    analysis on the audio which is constantly recorded.
    """
    with lr.RecAnalysis(config.REC_N, config.N_CHANNELS) as rec:
        rec.run()
    print("done analysis")


def ondemand_target():
    """target function for the multiprocessing.Process which will run
    analysis like onset quantization or BPM estimation on demand.
    """
    with lr.AnalysisOnDemand(config.REC_N, config.N_CHANNELS) as rec:
        rec.run()
    print("done ondemand")


if __name__ == "__main__":
    with lr.RecAudio(config.REC_N, config.N_CHANNELS) as rec:
        ap = Process(target=analysis_target)
        ap2 = Process(target=ondemand_target)
        ap.start()
        ap2.start()

        print(sd.query_devices())
        clave, _ = sf.read("../data/clave.wav", dtype=np.float32)
        clave = np.concatenate(
            (
                1 * clave[:, None],
                np.zeros((config.SR - len(clave), 1), dtype=np.float32),
            )
        )
        loop = Loop(rec, Audio(clave))
        loop.start()

        if config.HEADPHONE_DEVICE != config.DEVICE:
            hl = ExtraOutput(loop)

        # Some example effects that can be applied. TODO: make more
        # interesting/intuitive
        ps = pedalboard.PitchShift(semitones=-6)
        ds = pedalboard.Distortion(drive_db=20)
        delay = pedalboard.Delay(0.8, 0.1, 0.3)
        limiter = pedalboard.Limiter()
        loop.actions.append(
            Effect(0, 10000000, lambda x: limiter(x, config.SR))
        )

        midi = MIDIHandler(loop)

        plan_thread = threading.Thread(target=plan_callback, args=(loop,))
        plan_thread.start()
        ap.join()
        ap2.join()
        plan_thread.join()
        sd.sleep(10)
