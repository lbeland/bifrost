import numpy as np
import time
from collections import deque
from meegkit.phase import ECHT

seconds = 10
fs = 10000
dt = 1.0 / fs
emitted = 0

buffer = deque(maxlen=2000)

f0=10.0
filt_BW = f0 / 2
l_freq = f0 - filt_BW / 2
h_freq = f0 + filt_BW / 2
echt = ECHT(l_freq, h_freq, fs,filt_order=1)


def add_one(x):
    return x + 1

def tanh_sample(x):
    return np.tanh(x)

def echt_sample(x):
    return echt.fit_transform(x)

start = time.time()
for i in range(fs*seconds):
    sample = np.sin(2.0 * np.pi * f0 * (emitted * dt))

    emitted += 1

    buffer.append(sample)

    buffer_1 = add_one( np.array(buffer))
    buffer2 = tanh_sample(buffer_1)
    buffer3 = echt_sample(buffer2)

print(time.time() - start)
