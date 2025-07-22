import numpy as np
from scipy.signal import butter, lfilter

def filter(data, order=4, cutoff_freq=5, sampling_rate=200):
    '''The meaning of the parameters of butter:
    N=4 means the order of the filter
    Wn=5 means the cutoff frequency is 5Hz.
    fs=200 means the sampling frequency is 200Hz.'''
    b, a = butter(order, [cutoff_freq], fs=sampling_rate)
    filtered_data = lfilter(b, a, data)
    return filtered_data