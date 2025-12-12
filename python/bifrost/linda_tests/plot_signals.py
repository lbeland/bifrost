import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

file_echt_cali = 'sine.echt_cal.bin'
file_echt = 'sine.echt.bin'
file_orig = 'sine.orig.bin'
file_noisy = 'sine.noisy.bin'

phase_cali = np.fromfile(file_echt_cali, dtype=np.float32)
phase_uncali = np.fromfile(file_echt, dtype=np.float32)
data_orig = np.fromfile(file_orig, dtype=np.float32)
data_noisy = np.fromfile(file_noisy, dtype=np.float32)


win_shift = int(len(data_orig) / len(phase_cali))
win_size = 2000

# Calculate error
true_phase = np.angle(hilbert(data_orig))
time_values = list(range(win_size-1, len(data_orig), win_shift))
phase_cali = phase_cali[:len(time_values)]      # shorten at the end because of gulping
phase_uncali = phase_uncali[:len(time_values)]

error_cali = phase_cali - true_phase[time_values]
error_uncali = phase_uncali - true_phase[time_values]
print(f"Mean absolute phase error (calibrated): {np.mean(np.abs(error_cali))}")
print(f"Mean absolute phase error (uncalibrated): {np.mean(np.abs(error_uncali))}")

plt.plot(data_noisy, label='Signal')
plt.plot(time_values, phase_cali, '-o',label='ecHT phase calibrated')
plt.plot(time_values, phase_uncali, '-o',label='ecHT phase uncalibrated')
plt.plot(true_phase, label='True phase')
plt.grid(which='both')
plt.minorticks_on()
plt.xlabel('Sample Index')
plt.ylabel('Phase')

plt.legend()

plt.show()