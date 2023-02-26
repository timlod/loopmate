# Configuration for loopmate

sr = 96000
device = "default"
# Desired latency of audio interface, in ms
latency = 0.002
# Blocksize to use in processing, the lower the higher the CPU usage, and lower
# the latency
blocksize = 512
# Length (in ms) of blending window, e.g. used to remove pops in muting, or
# applying transformations to audio
blend_length = 0.005
