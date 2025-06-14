'''
Load Axivity data
- Dowanload [ORIGINAL](#https://github.com/ImperialCollegeLondon/ontrack-activity-classifier/tree/master/training_data) dataset (stored with Git LFS)
- Replace baseDir with your folder path
'''
import datetime
import joblib
import pandas as pd
import os
from joblib import Parallel, delayed
from itertools import chain
from tqdm import tqdm

from config import ax_data_dir, cache_dir
from config import WINDOW_SIZE, OVERLAPPING_PERCENTAGE, AW_SAMPLING_RATE

from dataTransformer.sliding_window import splitIntoOverlappingWindows

def load_ax_data():
    # Converting data files to pandas dataframes, we have a couple of different file formats for the recordings and these are labeled by device type in the CSV index
    def load_data(data_index_csv):
        dfs = {}

        def read_data_for_file(filepath, datestart, dateend, device, classification):
            '''
            Read data from a file and return a DataFrame.
            '''
            file_data = dfs.get(filepath)
            if file_data is None:
                file_data = pd.read_csv(ax_data_dir + '//' + filepath)
                ax_df = file_data
                ax_df.columns = ["Timestamp", "X","Y","Z"]
                ax_df["Timestamp"] = ax_df["Timestamp"].map(lambda t: datetime.datetime.fromtimestamp(t) if type(t) is float else t)
                ax_df = ax_df.set_index(pd.DatetimeIndex(ax_df['Timestamp']))
                ax_df = ax_df.drop("Timestamp", axis=1)
                dfs[filepath] = ax_df
            df = dfs[filepath][datestart:dateend].copy()
            df["Device"] = device
            df["Classification"] = classification
            return df
        
        
        def read_data_for_user(row):
            device = row["device"]
            if device == "ax":
                return read_data_for_file(row["filepath"], row["datestart"], row["dateend"], row["device"], row["classification"])
            else:
                raise Exception(f"Unknown Device {device}")
                
        accelerometer_data = data_index_csv.apply(read_data_for_user, axis=1)

        # Count the number of data
        data_amount_ax = datetime.timedelta(seconds = 0)
        for data in accelerometer_data:
            length = datetime.timedelta(seconds = (data.index.max() - data.index.min()).total_seconds())
            device = data.iloc[0]["Device"]
            if device == "ax":
                data_amount_ax += length
            else:
                raise Exception(f"Unknown Device {device}")


        print("[info@load_ax_data] -> Examples Axivity     " + str(data_amount_ax))
        return accelerometer_data

    checkpoint_file = os.path.join(cache_dir, 'AX.joblib')
    if os.path.exists(checkpoint_file):
        data = joblib.load(checkpoint_file)
        data_transport = data['transport']
        data_walking = data['walking']
        data_other = data['other']
        data_move = data['movement']
        print("[info@load_ax_data] -> Checkpoint loaded.")
    else:
        print("[info@load_ax_data] -> Checkpoint not found, loading data from CSV files...")
        data_transport = load_data(pd.read_csv(ax_data_dir + '/data_transport_index.csv'))
        data_walking = load_data(pd.read_csv(ax_data_dir + '/data_walking_index.csv'))
        data_other = load_data(pd.read_csv(ax_data_dir + '/data_other_index.csv', header=0))
        data_move = load_data(pd.read_csv(ax_data_dir + '/data_movement_index.csv'))

        # Save the checkpoint
        data = {
            'transport': data_transport,
            'walking': data_walking,
            'other': data_other,
            'movement': data_move
        }
        joblib.dump(data, checkpoint_file)
        print("[info@load_ax_data] -> Checkpoint saved.")
    
    data_move, data_transport, data_walking, data_other = splitAll(data_transport, data_walking, data_other, data_move)

    return data_move, data_transport, data_walking, data_other

def splitAll(data_transport, data_walking, data_other, data_move):
    chunkedTransport = Parallel(n_jobs=-1)(delayed(splitIntoOverlappingWindows)(x) for x in data_transport)
    # chunkedTransport0424 = Parallel(n_jobs=-1)(delayed(splitIntoOverlappingWindows)(x) for x in tqdm(data_transport_0424, desc="Chunking Transport data 0424"))
    chunkedWalking = Parallel(n_jobs=-1)(delayed(splitIntoOverlappingWindows)(x) for x in data_walking)
    chunkedOther = Parallel(n_jobs=-1)(delayed(splitIntoOverlappingWindows)(x) for x in data_other)
    chunkedMove = Parallel(n_jobs=-1)(delayed(splitIntoOverlappingWindows)(x) for x in data_move)

    flatMoveAX = list(chain.from_iterable(
        [list( chunkedMove[i] ) for i in range(len(chunkedMove))],
    ))

    flatWalkingAX = list(chain.from_iterable(
        [list( chunkedWalking[i] ) for i in range(len(chunkedWalking))],
    ))

    flatTransportAX = list(chain.from_iterable(
        [list( chunkedTransport[i] ) for i in range(len(chunkedTransport))],
    ))

    # flatTransport0424AX = list(chain.from_iterable(
    #     [list( chunkedTransport0424[i] ) for i in tqdm(range(len(chunkedTransport0424)),desc="Flatten Transport data 0424")]
    # ))

    flatOtherAX = list(chain.from_iterable(
        [list( chunkedOther[i] ) for i in range(len(chunkedOther))],
    ))

    return flatMoveAX, flatTransportAX, flatWalkingAX, flatOtherAX