from config import *

def splitIntoOverlappingWindows(df, ax_window_size=None, ax_step_size=None):
    """
    Splits a time-series DataFrame into overlapping windows.

    Parameters:
    - df (pd.DataFrame): The input time-series DataFrame.
    - window_size (int): Number of rows in each window.
    - step_size (int): Number of rows to shift the window for the next segment.

    Returns:
    - List[pd.DataFrame]: A list of DataFrame windows.
    """
    if not ax_window_size and not ax_step_size:
        ax_window_size = int(WINDOW_SIZE * AW_SAMPLING_RATE)
        ax_step_size = int(ax_window_size * OVERLAPPING_PERCENTAGE)

    # Drop unnecessary columns
    if "Device" in df.columns and "Classification" in df.columns:
        df = df.drop(["Device", "Classification"], axis=1)

    # print('-'*50)
    # print(df)
    # print('-'*50)

    # Resample to 20Hz (0.05 seconds interval)
    dfResample = df.resample("0.05s").mean().interpolate().dropna()

    # Create overlapping windows
    windows = [
        dfResample.iloc[i:i + ax_window_size]
        for i in range(0, len(dfResample) - ax_window_size + 1, ax_step_size)
    ]
    
    return windows