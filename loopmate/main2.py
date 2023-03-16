from __future__ import annotations

import asyncio
import time

import mido
import numpy as np
import pedalboard
import sounddevice as sd
import soundfile as sf

from loopmate import config
from loopmate.actions import (
    BackCaptureTrigger,
    Mute,
    MuteTrigger,
    RecordTrigger,
)
from loopmate.loop import Audio, Loop


class MidiQueue:
    def __init__(self, queue, loop):
        self.queue = queue
        self.loop = loop
        # self.aioloop = aioloop
        self.port = mido.open_input(callback=self.receive)

    def receive(self, message):
        if message.type == "note_on":
            t = time.time()
            # self.aioloop.call_soon_threadsafe(self.queue.put_nowait(message))
            print(time.time() - t)
            # self.queue.task_done()
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
                when = (
                    self.loop.anchor.loop_length
                    - round(self.loop.callback_time.output_delay * config.sr)
                ) % self.loop.anchor.loop_length
                self.loop.actions.actions.append(
                    RecordTrigger(
                        when,
                        self.loop.anchor.loop_length,
                        spawn=RecordTrigger(
                            when, self.loop.anchor.loop_length
                        ),
                    )
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


async def main():
    print(sd.query_devices())
    piano, _ = sf.read("../data/piano.wav", dtype=np.float32)
    clave, _ = sf.read("../data/clave.wav", dtype=np.float32)
    print(clave.shape)
    clave = np.concatenate(
        (
            1 * clave[:, None],
            np.zeros((config.sr - len(clave), 1), dtype=np.float32),
        )
    )
    # loop = Loop(Audio(piano))
    loop = Loop(Audio(clave))
    loop.start()
    # loop = Loop()
    print(loop)

    # TODO: plan!
    ps = pedalboard.PitchShift(semitones=-6)
    ds = pedalboard.Distortion(drive_db=20)
    delay = pedalboard.Delay(0.8, 0.1, 0.3)

    cq = asyncio.Queue()

    midi = MidiQueue(cq, loop)

    try:
        go = True
        while go:
            # print(cq.qsize())
            trigger = await loop.actions.plans.get()
            # trigger = asyncio.run_coroutine_threadsafe(
            #     loop.actions.plans.get(), aioloop
            # ).result()
            if isinstance(trigger, RecordTrigger):
                print(f"\rgot record in main")
                loop.record()
                continue
            elif isinstance(trigger, BackCaptureTrigger):
                loop.backcapture(trigger.n_loops)
                continue
    except (sd.CallbackStop, sd.CallbackAbort):
        print("Stopped")


if __name__ == "__main__":
    aioloop = asyncio.get_event_loop()
    aioloop.run_until_complete(main())

    # root = Tk()
    # frm = ttk.Frame(root, padding=10)
    # frm.grid()
    # ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
    # ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
    # root.mainloop()
