import numpy as np
from scipy.signal import welch

def preprocess_data(x, y, s, fs=128):
    f = []
    psd = []
    for ndarray in x:
        channel_f = []
        for channel in range(ndarray.shape[1]):
            frequency, powerSpectralDensity = welch(ndarray[:, channel], fs=fs, nperseg=fs * 2, noverlap=0.5)
            f = frequency
            channel_f.append(powerSpectralDensity)
        psd.append(np.array(channel_f).T)

    return f, psd, y, s
