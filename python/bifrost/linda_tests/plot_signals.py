import numpy as np
import matplotlib.pyplot as plt

file_echt = 'sine.echt.bin'
file_orig = 'sine.orig.bin'

data = np.fromfile(file_echt, dtype=np.float32)
data_orig = np.fromfile(file_orig, dtype=np.float32)

plt.plot(data_orig, label='Original Signal')
plt.plot(np.linspace(0,len(data_orig), len(data)), data, '-o',label='ecHT Processed Signal')

plt.legend()

plt.show()