import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

file_echt = 'sine.echt.bin'
file_orig = 'sine.orig.bin'
file_noisy = 'sine.noisy.bin'

data = np.fromfile(file_echt, dtype=np.float32)
data_orig = np.fromfile(file_orig, dtype=np.float32)
data_noisy = np.fromfile(file_noisy, dtype=np.float32)

plt.plot(data_noisy, label='Noisy Signal')
plt.plot(np.linspace(0,len(data_orig), len(data)), data, '-o',label='ecHT phase')
plt.plot(np.angle(hilbert(data_orig)), label='True phase without noise')

plt.legend()

plt.show()