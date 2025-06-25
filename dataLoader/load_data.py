import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to the Python path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_csv(file_path:str, useFilter=True) -> pd.DataFrame:
    """Parse a CSV file containing accelerometer data.
    return: DataFrame with columns 'Timestamp', 'X', 'Y', 'Z'."""
    ori_data = pd.read_csv(file_path)
    data = pd.DataFrame()
    data['Timestamp'] = pd.to_datetime(ori_data['Timestamp'], unit='s')
    data['X'] = ori_data['accelerationX']
    data['Y'] = ori_data['accelerationY']
    data['Z'] = ori_data['accelerationZ']
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data.set_index('Timestamp')
    data = data.dropna()  # Drop rows with NaN values
    if useFilter:
        from dataTransformer.filter import filter
        data["X"] = filter(data["X"])
        data["Y"] = filter(data["Y"])
        data["Z"] = filter(data["Z"])
    return data


def combine(data1:list, data2:list) -> np.array:
    if not data2:
        return np.array(data1)
    data1, data2 = np.array(data1), np.array(data2)
    return np.vstack([data1, data2])

def df2array(df:pd.DataFrame) -> np.array:
    return np.vstack([df.values for df in df])

def load_data(data_dir:str, useFilter=True, loadNewTransport=False) -> np.array:
    """Load CSV files from a directory and return a list of DataFrames.
    Set useFilter to True to apply filtering on the data."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")
    
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data_list = []
    
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        df = parse_csv(file_path, useFilter=useFilter)
        data_list.append(df)
    return data_list

def load_data_from_original_sources(loadNewTransport=False) -> tuple:
    """Load the data from original sources (AX and LAB data).
    Set loadNewTransport to True to include new transport data.
    Return type: list[pd.Dataframe]"""
    from config import ax_newT_xsl, ax_newT_csv
    from dataLoader.load_ax_data import load_ax_data
    from dataLoader.load_new_transport_data import load_new_transport_data
    from dataLoader.load_lab_data import load_lab_data

    ax_m, ax_t, ax_w, ax_o = load_ax_data()
    lab_m, lab_o = load_lab_data()
    if loadNewTransport:
        new_ax_m, new_ax_t, new_ax_w, new_ax_o = load_new_transport_data(excel_path=ax_newT_xsl, csvPath=ax_newT_csv)

    movement = combine(ax_m, lab_m)
    transport = combine(ax_t, None)
    walking = combine(ax_w, None)
    other = combine(ax_o, lab_o)
    return movement, transport, walking, other