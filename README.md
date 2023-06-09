# loopmate

`loopmate` is a looper/loop station built mainly for acoustic instruments
(although it will work with any audio input) with unlimited tracks, MIDI
control and event quantization.

`loopmate` is in an alpha stage - while basic functionality works, it may take
some effort and additional documentation to make it easy to work with.
Shamefully, `loopmate` is currently untested - mainly, because I'm not sure how
to set it up right, given that ultimately things depend on realtime audio
capture and multiple processes. There are classes and functions which could be
tested regardless, which should be prioritized at some point.

## Features
- unlimited tracks (as CPU and RAM permit - it should be more than enough)
  - arbitrary IO mapping given what your audio interface permits
- 'always-on' recording
  - `loopmate` always records the latest audio, keeping a set length of it in a
    circular audio array
  - if you liked what you played during the last loop, you can immediately add
    it as new loop audio with a single command - meaning you don't have to
    trigger new loops into the future
- near-realtime analysis of recorded audio
  - `loopmate` computes the STFT, an onset_envelope, and a tempogram in an online
    fashion
- Advanced/intelligent commands based on recorded audio
  - `loopmate` can detect onsets on recorded audio in a few hundred ms, or detect
    the BPM live as you go
    - Quantize your loop boundaries (when you start/end a loop) to musical
      onsets, or to the BPM of your performance
- FX can be applied to each recorded audio loop individually, or on top of the
  mix of all audio
- Have a separate headphone mix latency-aligned to your PA
  - possibility to mute the PA (not headphone mix) when recording a new loop
    for clean looping of mic'ed instruments

## Planned features

- MIDI beat clock sync
  - sync to other devices such that loops are aligned with the rest of your
    performance
- more advanced (stereo) output / mixing
  - currently, there is only naïve mono/stereo, that is, two inputs are sent to
    LR channels, one or more are mixed down to mono
- better UX accessing individual tracks/overdubs in an intuitive manner
- more features based on live analysis of the recorded audio, like capturing
  samples which can be triggered one-shot during live performance (as opposed
  to always looping)
- a dedicated device running `loopmate`, including a MIDI controller in the
  form of a Multipad, where commands are issued by striking pads

## Requirements

- Python 3.10
  - there are switch/case statements and type hints that require 3.10
- MIDI input
  - for a simple, keyboard-based test, you can use https://vmpk.sourceforge.io/

`loopmate` is tested on Linux, but should work on macOS and Windows as well, as
the audio processing is based on the multi-platform
https://github.com/spatialaudio/python-sounddevice.

See what dependencies need to be installed in [pyproject.toml](pyproject.toml).

## Setup

It is recommended to create a virtual environment to install this package into:
0. `python -m venv venv`
1. `pip install -e .` - makes a development install (code changes are reflected
   without reinstall)
2. To run, use `python loopmate/main.py`

Once `main` is running, `loopmate` will listen for input on all MIDI channels.
For a proper setup, you need to modify `config.py`.

## Configuration
In [config.py](loopmate/config.py) you can set things like input device, which
channels to record, the sampling rate, blocksize of the audio process, which
MIDI port/channel to listen on, and much more.

# Design/Architecture

Disclaimer: This is my first foray into realtime-constrained audio processing.
Since I have not looked at any serious alternatives (most open-source options I
found are either very simple or rather complex [and written in C++, which I'm
not very experienced in]). Therefore, there may be some questionable choices in
the architecture.

I did not intend functionality to grow this much. Had I anticipated this, I
might have written `loopmate` in another programming language. Python is what I
know well, so that's what I started out with.

The realtime audio is handled by
[python-sounddevice](https://github.com/spatialaudio/python-sounddevice).
sounddevice provides bindings for the multi-platform
[PortAudio](http://www.portaudio.com/) library. `loopmate` uses (one or two, in
case of an additional headphone output) sounddevice.Stream(s) with a callback
function that is executed at more or less consistent intervals. The callback
populates audio buffers which will be played back by PortAudio.

## Multiple processes

Python does not make it easy to use multiple cores in a simple way:
multi-threading happens on a single core only, and every thread may be impacted
by the Global Interpreter Lock (GIL). The GIL is the main reason why it's
usually not recommended to use Python for realtime audio - for example, if
garbage collection happens to lock the callback thread at the wrong moment,
this could lead to buffer underflow, and thus audio glitches.

To minimize load on the callback thread, `loopmate` uses two additional
processes, meaning there are 3 cores the program can utilize:
1. Main process, contains:
   - high-priority audio thread
   - MIDI input handling
2. RT Analysis thread
   - computes STFT/onset envelope/tempogram on each incoming audio buffer
3. On-demand analysis thread
   - computes onsets, BPM estimate, etc. when needed

The processes all share memory to minimize latency. IPC latency is currently
down to about 0.1ms.

## Actions & triggers

Each recorded audio loop, as well as the sum of all loops, contain a number of
'action queues'. Actions can span multiple consecutive audio buffers, and thus
allow for smooth transitions of audio, e.g. when muting, or applying FX. See
[actions.py](loopmate/actions.py) for the implementation.

There are also triggers, which can be used to schedule actions and commands
(like initiating recording) at a set time.

Actions and triggers can be set to work based on MIDI commands, or spawned
regularly, once every loop iteration.

## Looped audio

`loopmate` uses the concept of an 'anchor loop' - this will be the first loop
that you record. The anchor defines the rhythm of the loop. However,
consecutive recordings don't have to take the same amount of time as the anchor
loop. Shorter overdubs work as expected, and longer additional tracks are
extended such that their length is a power of 2 times the anchor loop's length.

For example, if you start with a loop of 4s, and then start recording 6s of
audio on the 1, the second loop will be extended to 8s (effectively double the
anchor loop).

If you don't start or stop recording with perfect timing, don't worry: `loopmate`
will quantize subsequent loops to the anchor loop's boundaries. Since the
recording is 'always on', you will never lose audio you played shortly before
you hit 'record'.
