# Configuration for loopmate

# Global sample rate for all audio being played/recorded - if loading different
# SR files, convert them
sr = 44100
channels = 1
device = "default"
# device = "USB AUDIO CODEC"
device = "Steinberg UR22: USB Audio"
# device = "Steinberg UR22 Analogue Stereo"
# Desired latency of audio interface, in ms
latency = 0.001
# Blocksize to use in processing, the lower the higher the CPU usage, and lower
# the latency
blocksize = 256
# Length (in ms) of blending window, e.g. used to remove pops in muting, or
# applying transformations to audio
blend_length = 0.05
quantize_ms = 0.2
# Output delay from speaker sound travel
air_delay = 0.0
# Maximum recording length (in seconds). Will constantly keep a buffer of sr *
# this many samples to query backwards from.
max_recording_length = 60

blend_frames = round(sr * blend_length)
