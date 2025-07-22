'''
- Load new transport data collected on 24th April 2025
- Data collected using Apple Watch (Motion Tracking)
'''
import datetime
import pandas as pd
import os
import sys
from loguru import logger
from joblib import Parallel, delayed
from itertools import chain

# Configure loguru logger with colors
logger.remove()  # Remove default handler
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True
)
logger.add(
    "logs/load_new_transport_data_{time:YYYY-MM-DD}.log", 
    rotation="1 day", 
    retention="7 days", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    colorize=False
)

# 自定义级别颜色
logger.level("INFO", color="<blue>")
logger.level("SUCCESS", color="<green>")
logger.level("WARNING", color="<yellow>")
logger.level("ERROR", color="<red>")
logger.level("DEBUG", color="<cyan>")

# Add the parent directory to the Python path to import config and sliding window transformer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataTransformer.sliding_window import splitIntoOverlappingWindows

def load_new_transport_data(basePath, excel_path=None, csvPath=None):
    def load_data_0424(data_index_csv):
        dfs = {}

        def read_data_for_file(filepath, datestart, dateend, device, classification):
            file_data = dfs.get(filepath)
            if file_data is None:
                file_path = os.path.join(basePath, filepath)
                file_data = pd.read_csv(file_path)

                # Check if the file is empty
                if file_data.empty:
                    logger.warning(f"The file {file_path} is empty.")
                    return pd.DataFrame()

                ax_df = file_data[["Timestamp", "accelerationX", "accelerationY", "accelerationZ"]].copy()
                ax_df.columns = ["Timestamp", "X", "Y", "Z"]
                ax_df["Timestamp"] = ax_df["Timestamp"].map(lambda t: float(t) if isinstance(t, str) else t)
                ax_df["Timestamp"] = ax_df["Timestamp"].map(lambda t: datetime.datetime.fromtimestamp(t) if isinstance(t, (float, int)) else t)
                ax_df = ax_df.set_index(pd.DatetimeIndex(ax_df['Timestamp']))
                ax_df = ax_df.drop("Timestamp", axis=1)
                dfs[filepath] = ax_df

            datestart = datetime.datetime.fromtimestamp(datestart) # Convert string to datetime
            dateend = datetime.datetime.fromtimestamp(dateend)

            df = dfs[filepath][datestart:dateend].copy()
            df["Device"] = device
            df["Classification"] = classification

            return df

        def read_data_for_user(row):
            return read_data_for_file(row["filepath"], row["datestart"], row["dateend"], row["device"], row["classification"])

        accelerometer_data = data_index_csv.apply(read_data_for_user, axis=1)

        # Count the number of data
        data_amount_ax = datetime.timedelta(seconds = 0)
        for data in accelerometer_data:
            length = datetime.timedelta(seconds = (data.index.max() - data.index.min()).total_seconds())
            data_amount_ax += length
        logger.info(f"Length of New Transport data: {str(data_amount_ax)}")

        return accelerometer_data

    def process_transport_data(data_list):
        """Process the raw transport data by applying sliding windows and converting to numpy array format"""
        if data_list is None or len(data_list) == 0:
            logger.warning("No transport data to process")
            return None
            
        # Filter out empty DataFrames
        valid_data = [df for df in data_list if not df.empty]
        
        if not valid_data:
            logger.warning("All transport data DataFrames are empty")
            return None
            
        logger.info(f"Processing {len(valid_data)} transport data segments...")
        
        # Apply sliding window to each DataFrame
        chunked_transport = Parallel(n_jobs=-1)(delayed(splitIntoOverlappingWindows)(df[['X', 'Y', 'Z']]) for df in valid_data)
        
        # Flatten the results
        flat_transport = list(chain.from_iterable(
            [list(chunked_transport[i]) for i in range(len(chunked_transport))]
        ))
        
        logger.success(f"Processed {len(flat_transport)} transport data windows")
        return flat_transport

    # import csv data from excel file
    if excel_path or csvPath:
        if csvPath and os.path.exists(csvPath):
            csv_transport_0424 = pd.read_csv(csvPath)
            data_transport_0424 = load_data_0424(csv_transport_0424)
            # Process the data to match the expected format
            processed_data = process_transport_data(data_transport_0424)
            return processed_data
        if excel_path and not os.path.exists(csvPath):
            df = pd.read_excel(excel_path, sheet_name='Sheet1', header=0)
            baseDir = os.path.dirname(excel_path) # The same directory of excel_path
            savedPath = os.path.join(baseDir, "data_transport_index_0424.csv")
            df.to_csv(savedPath, index=False, float_format='%.4f')
            logger.success("Excel file converted to CSV.")
    else:
        raise ValueError("Either excel_path or csvPath must be provided.")
    
    csv_transport_0424 = pd.read_csv(csvPath)
    data_transport_0424 = load_data_0424(csv_transport_0424)
    # Process the data to match the expected format
    processed_data = process_transport_data(data_transport_0424)
    return processed_data


if __name__ == "__main__":
    # debug
    rootDir = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project'
    basePath = os.path.join(rootDir, "Data/MTWO_transport_0424")
    ax_newT_xsl = os.path.join(basePath, r"data_transport_index_0424.xlsx")
    ax_newT_csv = os.path.join(basePath, r"data_transport_index_0424.csv")

    data = load_new_transport_data(basePath=basePath, excel_path=ax_newT_xsl, csvPath=ax_newT_csv)

    # Print the first few rows of the loaded data
    for df in data:
        print(df.head())