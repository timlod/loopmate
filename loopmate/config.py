# Configuration for loopmate

# Global sample rate for all audio being played/recorded - if loading different
# SR files, convert them
sr = 48000
device = "default"
device = "USB AUDIO CODEC"
# Desired latency of audio interface, in ms
latency = 0.002
# Blocksize to use in processing, the lower the higher the CPU usage, and lower
# the latency
blocksize = 256
# Length (in ms) of blending window, e.g. used to remove pops in muting, or
# applying transformations to audio
blend_length = 0.005
quantize_ms = 0.2
