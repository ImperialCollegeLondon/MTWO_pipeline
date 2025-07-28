import numpy as np
from scipy.signal import butter, lfilter

def filter(data, order=4, low_cutoff=0.5, high_cutoff=5, sampling_rate=20):
    '''
    带通滤波器，去除重力分量（低频）和高频噪声。
    参数说明：
    order: 滤波器阶数
    low_cutoff: 带通低频截止（Hz），用于去除重力分量
    high_cutoff: 带通高频截止（Hz），用于去除高频噪声
    sampling_rate: 采样率（Hz）
    '''
    b, a = butter(order, [low_cutoff, high_cutoff], btype='bandpass', fs=sampling_rate)
    filtered_data = lfilter(b, a, data)
    return filtered_data