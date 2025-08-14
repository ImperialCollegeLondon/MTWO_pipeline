import os
import re
import pandas as pd
import sys

# Add the parent directory to the Python path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import save_dir

from loguru import logger
# Configure loguru logger with colors
logger.remove()  # Remove default handler
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

def get_gerf_files(data_dir, pattern_style='gerf'):
    '''Load all GERF data files from the specified directory.
    Args:
        data_dir (str): The directory containing the GERF data files.
        pattern_style (str): The style of the file name pattern to match. 
                             Default is 'gerf', which matches GERF files.
                             Other options include 'all_csv' for all CSV files or a custom regex pattern.
    Returns:
        file_list: A list of file names that match the specified pattern.
    '''
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")
    
    if pattern_style == 'gerf':
        pattern = re.compile(r'^GERF-(L|R)-[A-Z]\d{3}-M(1|2|3|6)-S\d{4}\.csv$')
    elif pattern_style == 'gerf_mo':
        pattern = re.compile(r'^GERF-(L|R)-[A-Z]\d{3}-M(1|3)-S\d{4}\.csv$')
    elif pattern_style == 'csv':
        pattern = re.compile(r'.*\.csv$')
    else:
        pattern = re.compile(pattern_style)

    file_list = []
    for _file in os.listdir(data_dir):
        if pattern.match(_file):
            file_list.append(_file)
    return file_list

def init_res_csv(data_dir, gt='default'):
    res_csv = os.path.join(save_dir, "res_summary.csv")
    file_list = get_gerf_files(data_dir)
    columns=["Sample", "Ground Truth", "xgboost", "svm", "knn", "rf", "lr", "mlp"]
    rows = []
    for file_name in file_list:
        sample_name = os.path.splitext(file_name)[0]

        if gt == 'default':
            # Default ground truth mapping
            definition = {
                1: "Movement",
                2: "Walking",
                3: "Others",
                6: "Transport",
                # 4: "CHECK"
            }
            ground_truth = definition[int(sample_name.split("-")[3][-1])] # 这里根据需要更改
        else:
            ground_truth = 'Movement'

        new_row = {
            "Sample": sample_name,
            "Ground Truth": ground_truth,
            "xgboost": "",
            "svm": "",
            "knn": "",
            "rf": "",
            "lr": "",
            "mlp": ""
        }
        rows.append(new_row)
    # Make sure all columns are of string type
    df = pd.DataFrame(rows, columns=columns)
    for col in columns:
        df[col] = df[col].astype(str)

    df.to_csv(res_csv, index=False)
    logger.info(f"Results CSV initialised at {save_dir} as res_summary.csv.")