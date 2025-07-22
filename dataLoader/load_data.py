import numpy as np
import pandas as pd
import os
import sys
from config import getLogger
logger = getLogger()
# Add the parent directory to the Python path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def parse_csv(file_path:str, useFilter=True) -> pd.DataFrame:
    """Parse a CSV file containing accelerometer data.
    return: DataFrame with columns 'Timestamp', 'X', 'Y', 'Z'."""
    ori_data = pd.read_csv(file_path)
    data = pd.DataFrame()
    try:
        data['Timestamp'] = pd.to_datetime(ori_data['Timestamp'], unit='s')
        data['X'] = ori_data['accelerationX']
        data['Y'] = ori_data['accelerationY']
        data['Z'] = ori_data['accelerationZ']
    except KeyError:
        try:
            data['Timestamp'] = pd.to_datetime(ori_data.index, unit='s')
            data['X'] = ori_data['AccelX']
            data['Y'] = ori_data['AccelY']
            data['Z'] = ori_data['AccelZ']
        except KeyError:
            data['Timestamp'] = pd.to_datetime(ori_data['filteredTimesStrings'])
            data['X'] = ori_data['filteredData_1']
            data['Y'] = ori_data['filteredData_2']
            data['Z'] = ori_data['filteredData_3']

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
    # if not data2:
    if data2 is None or len(data2) == 0:
        logger.debug(f"data2 is None or empty, returning data1 as numpy array")
        if data1 is None:
            return None
        return np.array(data1)
    
    logger.debug(f"Before conversion - data1 type: {type(data1)}, data2 type: {type(data2)}")
    if data1 is not None:
        logger.debug(f"data1 length: {len(data1)}")
        if len(data1) > 0:
            logger.debug(f"data1[0] type: {type(data1[0])}, shape: {data1[0].shape if hasattr(data1[0], 'shape') else 'N/A'}")
    if data2 is not None:
        logger.debug(f"data2 length: {len(data2)}")
        if len(data2) > 0:
            logger.debug(f"data2[0] type: {type(data2[0])}, shape: {data2[0].shape if hasattr(data2[0], 'shape') else 'N/A'}")
    
    data1, data2 = np.array(data1), np.array(data2)
    logger.debug(f"After conversion - data1 shape: {data1.shape}, data2 shape: {data2.shape}")
    
    try:
        combined_data = np.vstack([data1, data2])
    except Exception as e:
        logger.error(f"Error combining data: {e}")
        logger.error(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}")
        logger.error(f"Data1: {data1[:5]}")  # Log first 5 rows of data1
        logger.error(f"Data2: {data2[:5]}")
        return None
    return combined_data

def df2array(df:pd.DataFrame) -> np.array:
    return np.vstack([df.values for df in df])

def load_data(data_dir:str, useFilter=True) -> np.array:
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
    from config import basePath, ax_newT_xsl, ax_newT_csv
    from dataLoader.load_ax_data import load_ax_data
    from dataLoader.load_new_transport_data import load_new_transport_data
    from dataLoader.load_lab_data import load_lab_data

    ax_m, ax_t, ax_w, ax_o = load_ax_data()
    lab_m, lab_o = load_lab_data()

    new_transport_data = None
    if loadNewTransport:
        new_transport_data = load_new_transport_data(basePath=basePath, excel_path=ax_newT_xsl, csvPath=ax_newT_csv)
        if new_transport_data is not None:
            logger.success("New transport data loaded successfully.")
        else:
            logger.warning("No new transport data found or loaded.")
            
    movement = combine(ax_m, lab_m)
    transport = combine(ax_t, new_transport_data)
    logger.debug(f"Before combine new_transport_data: ax_t length = {len(ax_t) if ax_t is not None else 'None'}")
    logger.debug(f"new_transport_data length = {len(new_transport_data) if new_transport_data is not None else 'None'}")
    logger.debug(f"After combine new_transport_data: transport length = {len(transport) if transport is not None else 'None'}")
    walking = combine(ax_w, None)
    other = combine(ax_o, lab_o)

    return movement, transport, walking, other