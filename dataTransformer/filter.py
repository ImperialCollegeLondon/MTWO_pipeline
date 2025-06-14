import numpy as np
from scipy.signal import butter, lfilter

def butterworth_filter(data, cutoff=3.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def filter(data, threshold=0.3):
    data = butterworth_filter(data)
    # Divide by two if the absolute value of acceleration is greater than threshold
    # data = np.where(data > threshold, data / 2, data)
    # data = np.where(data < -threshold, data / 2, data)
    return data
