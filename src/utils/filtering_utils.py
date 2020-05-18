import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, periodogram, hilbert, savgol_filter


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    def butter_bandpass(lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, lowcut, fs, order=2):
    def butter_lowpass(lowcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = butter(order, low, btype='lowpass')
        return b, a

    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def get_psd(t, signal):
    fs = 1.0 / (t[2] - t[1])
    f, psd = periodogram(signal, fs)
    return f, psd

def get_cutoff_freqz(f, psd, width):
    threshold = np.quantile(psd, 0.95)
    peaks, info = find_peaks(psd, threshold=threshold, prominence=1)
    # sort peaks by prominence
    prominences = info["prominences"]
    tmp = list(zip(peaks, prominences))
    tmp.sort(key=lambda a: a[1])
    peaks, prominences = list(zip(*tmp))
    highest_peak = peaks[-1]

    f_low = np.maximum(0, f[highest_peak] - width)
    f_high = f[highest_peak] + width
    return f_low, f_high

def scale(signal):
    return (signal - np.min(signal))/(np.max(signal) - np.min(signal))