# Global configuration for loopmate
from math import ceil

# Global sample rate for all audio being played/recorded - if loading different
# SR files, convert them
SR = 48000
CHANNELS = 1
# TODO: allow configuration of this to not necessarily always record everything
RECORD_CHANNELS = [0, 1]
DEVICE = "default"
# Desired latency of audio interface, in ms
LATENCY = 0.001
# Blocksize to use in processing, the lower the higher the CPU usage, and lower
# the latency
BLOCKSIZE = 128
# Length (in ms) of blending window, e.g. used to remove pops in muting, or
# applying transformations to audio
BLEND_LENGTH = 0.05
QUANTIZE_MS = 0.2
# Output delay from speaker sound travel
AIR_DELAY = 0.0
# Maximum recording length (in seconds). Will constantly keep a buffer of sr *
# this many samples to query backwards from.
MAX_RECORDING_LENGTH = 60
# MIDI (output) port to use as MIDI input
MIDI_PORT = 0
MIDI_CHANNEL = 0

# STFT config
N_FFT = 2048
HOP_LENGTH = BLOCKSIZE
TG_WIN_LENGTH = 1024
TG_PAD = 2 * TG_WIN_LENGTH - 1

REC_N = MAX_RECORDING_LENGTH * SR
BLEND_SAMPLES = round(SR * BLEND_LENGTH)

# Parameters for onset detection. See
# https://librosa.org/doc/latest/generated/librosa.util.peak_pick.html#librosa.util.peak_pick
# for details
PRE_MAX = int(0.03 * SR // HOP_LENGTH)
POST_MAX = int(0.0 * SR // HOP_LENGTH + 1)
MAX_LENGTH = PRE_MAX + POST_MAX
MAX_ORIGIN = ceil(0.5 * (PRE_MAX - POST_MAX))
MAX_OFFSET = (MAX_LENGTH // 2) - MAX_ORIGIN

PRE_AVG = int(0.1 * SR // HOP_LENGTH)
POST_AVG = int(0.1 * SR // HOP_LENGTH + 1)
AVG_LENGTH = PRE_AVG + POST_AVG
AVG_ORIGIN = ceil(0.5 * (PRE_AVG - POST_AVG))
AVG_OFFSET = (AVG_LENGTH // 2) - AVG_ORIGIN

WAIT = int(0.03 * SR // HOP_LENGTH)
DELTA = 0.07

ONSET_DET_OFFSET = AVG_OFFSET if AVG_OFFSET > MAX_OFFSET else MAX_OFFSET
