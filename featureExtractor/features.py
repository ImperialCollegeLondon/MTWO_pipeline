import numpy as np
import pywt
'''
This module contains functions to compute features from accelerometer data.
Here the df refers to a single window of accelerometer data.
'''
def compute_features_MO(df, X="X", Y="Y", Z="Z"):
    # Compute acceleration magnitude
    df["Magnitude"] = np.sqrt(df[X]**2 + df[Y]**2 + df[Z]**2) - 1
    
    # Compute Jerk (Rate of Change of Acceleration)
    df["Jerk"] = df["Magnitude"].diff().fillna(0)

    # Compute entropy (captures randomness in motion)
    def entropy(signal):
        hist, _ = np.histogram(signal, bins=10, density=True)
        hist = hist[hist > 0] # Remove zero values
        return -np.sum(hist * np.log(hist))
 
    # Extract Features
    features = [
        df["Magnitude"].mean(),   # Mean acceleration
        df["Magnitude"].min(),    # Minimum acceleration
        df["Magnitude"].max(),    # Maximum acceleration
        df["Magnitude"].std(),    # Standard deviation
        # df[X].corr(df[Y]),       # Correlation XY 0
        # df[X].corr(df[Z]),       # Correlation XZ 0
        # df[Y].corr(df[Z]),        # Correlation YZ 0

        # np.sqrt(np.mean(df["Magnitude"]**2)),  # Root Mean Square (RMS) 0
        np.sum(df["Magnitude"]**2),  # Signal Energy
        entropy(df["Magnitude"]),    # Entropy
        df["Jerk"].mean(),           # Mean Jerk
        df["Jerk"].std(),            # Jerk Variability
    ]

    # -------------------------------------------------------------
    # Frequency domain features using FFT

    freqs = np.fft.fftfreq(len(df), d=0.05) # frequency bins for 20Hz sampling rate
    X_fft = np.abs(np.fft.fft(df[X].values)) # FFT for X-axis
    Y_fft = np.abs(np.fft.fft(df[Y].values))
    Z_fft = np.abs(np.fft.fft(df[Z].values))
    Mag_fft = np.abs(np.fft.fft(df["Magnitude"].values)) # FFT for Magnitude
    
    # Keep only the frequency components from 0-10Hz (relevant for human activities)
    idx = np.where((freqs >= 0) & (freqs <= 10))[0]
    
    features.extend([
        np.mean(X_fft[idx]),  # Average FFT magnitude on X axis
        # np.max(X_fft[idx]),   # X axis peak magnitude
        # np.argmax(X_fft[idx]),# X axis peak index
        # freqs[np.argmax(X_fft)], # Main frequency for X axis 这个玩意加上去是副作用
        
        np.mean(Y_fft[idx]),  # Y axis 1
        # np.max(Y_fft[idx]),
        # np.argmax(Y_fft[idx]),
        # freqs[np.argmax(Y_fft)],

        np.mean(Z_fft[idx]),  # Z axis 1
        # np.max(Z_fft[idx]),
        # np.argmax(Z_fft[idx]),
        # freqs[np.argmax(Z_fft)],

        # np.mean(Mag_fft[idx]), # Resultant magnitude
        # np.max(Mag_fft[idx]),
        # np.argmax(Mag_fft[idx]),
        # freqs[idx][np.argmax(Mag_fft[idx])]
    ])

    # -------------------------------------------------------------
    # Wavelet features using Discrete Wavelet Transform (DWT)

    xca3, xcd3, xcd2, xcd1 = pywt.wavedec(df[X], 'db1', level=3)
    yca3, ycd3, ycd2, ycd1 = pywt.wavedec(df[Y], 'db1', level=3)
    zca3, zcd3, zcd2, zcd1 = pywt.wavedec(df[Z], 'db1', level=3)

    features.extend([
        df[X].mean(),   # Mean X 1
        df[Y].mean(),   # Mean Y 1
        df[Z].mean(),   # Mean Z 1
        # df[X].std(),    # Std X
        # df[Y].std(),    # Std Y
        # df[Z].std(),    # Std Z
        # df[X].min(),    # Min X
        # df[Y].min(),    # Min Y
        # df[Z].min(),    # Min Z
        # df[X].max(),    # Max X
        # df[Y].max(),    # Max Y
        # df[Z].max(),    # Max Z
        # # peak-to-valley
        # df[X].max() - df[X].min(),
        # df[Y].max() - df[Y].min(),
        # df[Z].max() - df[Z].min(),
        # df['Magnitude'].max() - df['Magnitude'].min(),
        # wavelet features
        # xca3.mean(), xca3.std(),
        # np.sum(xcd3**2), np.sum(xcd2**2), np.sum(xcd1**2),
        # yca3.mean(), yca3.std(),
        # np.sum(ycd3**2), np.sum(ycd2**2), np.sum(ycd1**2),
        # zca3.mean(), zca3.std(),
        # np.sum(zcd3**2), np.sum(zcd2**2), np.sum(zcd1**2),
    ])

    return np.array(features)

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def compute_features(df, X="X", Y="Y", Z="Z"):
    # Compute acceleration magnitude
    df["Magnitude"] = np.sqrt(df[X]**2 + df[Y]**2 + df[Z]**2) - 1
    
    # Compute Jerk (Rate of Change of Acceleration)
    df["Jerk"] = df["Magnitude"].diff().fillna(0)

    # Compute entropy (captures randomness in motion)
    def entropy(signal):
        hist, _ = np.histogram(signal, bins=10, density=True)
        hist = hist[hist > 0] # Remove zero values
        return -np.sum(hist * np.log(hist))
 
    # Extract Features
    features = [
        df["Magnitude"].mean(),   # Mean acceleration
        df["Magnitude"].min(),    # Minimum acceleration
        df["Magnitude"].max(),    # Maximum acceleration
        df["Magnitude"].std(),    # Standard deviation
        df[X].corr(df[Y]),       # Correlation XY
        df[X].corr(df[Z]),       # Correlation XZ
        df[Y].corr(df[Z]),        # Correlation YZ

        np.sqrt(np.mean(df["Magnitude"]**2)),  # Root Mean Square (RMS)
        np.sum(df["Magnitude"]**2),  # Signal Energy
        entropy(df["Magnitude"]),    # Entropy
        df["Jerk"].mean(),           # Mean Jerk
        df["Jerk"].std(),            # Jerk Variability
    ]

    # -------------------------------------------------------------
    # Frequency domain features using FFT

    freqs = np.fft.fftfreq(len(df), d=0.05) # frequency bins for 20Hz sampling rate
    X_fft = np.abs(np.fft.fft(df[X].values)) # FFT for X-axis
    Y_fft = np.abs(np.fft.fft(df[Y].values))
    Z_fft = np.abs(np.fft.fft(df[Z].values))
    Mag_fft = np.abs(np.fft.fft(df["Magnitude"].values)) # FFT for Magnitude
    
    # Keep only the frequency components from 0-10Hz (relevant for human activities)
    idx = np.where((freqs >= 0) & (freqs <= 10))[0]
    
    features.extend([
        np.mean(X_fft[idx]),  # Average FFT magnitude on X axis
        # np.max(X_fft[idx]),   # X axis peak magnitude
        # np.argmax(X_fft[idx]),# X axis peak index
        # freqs[np.argmax(X_fft)], # Main frequency for X axis 这个玩意加上去是副作用
        
        # np.mean(Y_fft[idx]),  # Y axis
        # np.max(Y_fft[idx]),
        # np.argmax(Y_fft[idx]),
        # freqs[np.argmax(Y_fft)],

        # np.mean(Z_fft[idx]),  # Z axis
        # np.max(Z_fft[idx]),
        # np.argmax(Z_fft[idx]),
        # freqs[np.argmax(Z_fft)],

        # np.mean(Mag_fft[idx]), # Resultant magnitude
        # np.max(Mag_fft[idx]),
        # np.argmax(Mag_fft[idx]),
        # freqs[idx][np.argmax(Mag_fft[idx])]
    ])

    # -------------------------------------------------------------
    # Wavelet features using Discrete Wavelet Transform (DWT)

    xca3, xcd3, xcd2, xcd1 = pywt.wavedec(df[X], 'db1', level=3)
    yca3, ycd3, ycd2, ycd1 = pywt.wavedec(df[Y], 'db1', level=3)
    zca3, zcd3, zcd2, zcd1 = pywt.wavedec(df[Z], 'db1', level=3)

    features.extend([
        # df[X].mean(),   # Mean X
        # df[Y].mean(),   # Mean Y
        # df[Z].mean(),   # Mean Z
        # df[X].std(),    # Std X
        # df[Y].std(),    # Std Y
        # df[Z].std(),    # Std Z
        # df[X].min(),    # Min X
        # df[Y].min(),    # Min Y
        # df[Z].min(),    # Min Z
        # df[X].max(),    # Max X
        # df[Y].max(),    # Max Y
        # df[Z].max(),    # Max Z
        # # peak-to-valley
        # df[X].max() - df[X].min(),
        # df[Y].max() - df[Y].min(),
        # df[Z].max() - df[Z].min(),
        # df['Magnitude'].max() - df['Magnitude'].min(),
        # wavelet features
        # xca3.mean(), xca3.std(),
        # np.sum(xcd3**2), np.sum(xcd2**2), np.sum(xcd1**2),
        # yca3.mean(), yca3.std(),
        # np.sum(ycd3**2), np.sum(ycd2**2), np.sum(ycd1**2),
        # zca3.mean(), zca3.std(),
        # np.sum(zcd3**2), np.sum(zcd2**2), np.sum(zcd1**2),
    ])

    return np.array(features)

def compute_features_1(df, X="X", Y="Y", Z="Z"):
    # Compute acceleration magnitude
    df["Magnitude"] = np.sqrt(df[X]**2 + df[Y]**2 + df[Z]**2)
    
    # Compute Jerk (Rate of Change of Acceleration)
    df["Jerk"] = df["Magnitude"].diff().fillna(0)

    # Compute entropy (captures randomness in motion)
    def entropy(signal):
        hist, _ = np.histogram(signal, bins=10, density=True)
        hist = hist[hist > 0] # Remove zero values
        return -np.sum(hist * np.log(hist))
 
    # Extract Features
    features = [
        df["Magnitude"].mean(),   # Mean acceleration
        df["Magnitude"].min(),    # Minimum acceleration
        df["Magnitude"].max(),    # Maximum acceleration
        df["Magnitude"].std(),    # Standard deviation
        df[X].corr(df[Y]),       # Correlation XY
        df[X].corr(df[Z]),       # Correlation XZ
        df[Y].corr(df[Z]),        # Correlation YZ

        np.sqrt(np.mean(df["Magnitude"]**2)),  # Root Mean Square (RMS)
        np.sum(df["Magnitude"]**2),  # Signal Energy
        entropy(df["Magnitude"]),    # Entropy
        df["Jerk"].mean(),           # Mean Jerk
        df["Jerk"].std(),            # Jerk Variability
    ]

    # -------------------------------------------------------------
    # Frequency domain features using FFT

    freqs = np.fft.fftfreq(len(df), d=0.05) # frequency bins for 20Hz sampling rate
    X_fft = np.abs(np.fft.fft(df[X].values)) # FFT for X-axis
    Y_fft = np.abs(np.fft.fft(df[Y].values))
    Z_fft = np.abs(np.fft.fft(df[Z].values))
    Mag_fft = np.abs(np.fft.fft(df["Magnitude"].values)) # FFT for Magnitude
    
    # Keep only the frequency components from 0-10Hz (relevant for human activities)
    idx = np.where((freqs >= 0) & (freqs <= 10))[0]
    
    features.extend([
        np.mean(X_fft[idx]),  # Average FFT magnitude on X axis
        # np.max(X_fft[idx]),   # X axis peak magnitude
        # np.argmax(X_fft[idx]),# X axis peak index
        # freqs[np.argmax(X_fft)], # Main frequency for X axis 这个玩意加上去是副作用
        
        # np.mean(Y_fft[idx]),  # Y axis
        # np.max(Y_fft[idx]),
        # np.argmax(Y_fft[idx]),
        # freqs[np.argmax(Y_fft)],

        # np.mean(Z_fft[idx]),  # Z axis
        # np.max(Z_fft[idx]),
        # np.argmax(Z_fft[idx]),
        # freqs[np.argmax(Z_fft)],

        # np.mean(Mag_fft[idx]), # Resultant magnitude
        # np.max(Mag_fft[idx]),
        # np.argmax(Mag_fft[idx]),
        # freqs[idx][np.argmax(Mag_fft[idx])]
    ])

    # -------------------------------------------------------------
    # Wavelet features using Discrete Wavelet Transform (DWT)

    xca3, xcd3, xcd2, xcd1 = pywt.wavedec(df[X], 'db1', level=3)
    yca3, ycd3, ycd2, ycd1 = pywt.wavedec(df[Y], 'db1', level=3)
    zca3, zcd3, zcd2, zcd1 = pywt.wavedec(df[Z], 'db1', level=3)

    features.extend([
        # df[X].mean(),   # Mean X
        # df[Y].mean(),   # Mean Y
        # df[Z].mean(),   # Mean Z
        # df[X].std(),    # Std X
        # df[Y].std(),    # Std Y
        # df[Z].std(),    # Std Z
        # df[X].min(),    # Min X
        # df[Y].min(),    # Min Y
        # df[Z].min(),    # Min Z
        # df[X].max(),    # Max X
        # df[Y].max(),    # Max Y
        # df[Z].max(),    # Max Z
        # # peak-to-valley
        # df[X].max() - df[X].min(),
        # df[Y].max() - df[Y].min(),
        # df[Z].max() - df[Z].min(),
        # df['Magnitude'].max() - df['Magnitude'].min(),
        # wavelet features
        # xca3.mean(), xca3.std(),
        # np.sum(xcd3**2), np.sum(xcd2**2), np.sum(xcd1**2),
        # yca3.mean(), yca3.std(),
        # np.sum(ycd3**2), np.sum(ycd2**2), np.sum(ycd1**2),
        # zca3.mean(), zca3.std(),
        # np.sum(zcd3**2), np.sum(zcd2**2), np.sum(zcd1**2),
    ])

    return np.array(features)


if __name__ == "__main__":
    # Get the length of features
    import pandas as pd
    data = {
        "X": np.random.rand(10),
        "Y": np.random.rand(10),
        "Z": np.random.rand(10)
    }
    df = pd.DataFrame(data)
    features = compute_features(df)
    print(f"\nLength of features : {len(features)}\n")