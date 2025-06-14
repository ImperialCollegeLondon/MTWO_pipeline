'''
- Load the lab data collected using Vicon
'''
import pandas as pd
import os
import re
import joblib
from joblib import Parallel, delayed
from pathlib import Path
from itertools import chain
import gc
from tqdm import tqdm

from config import WINDOW_SIZE, OVERLAPPING_PERCENTAGE, LAB_SAMPLING_RATE, lab_data_dir, cache_dir

def annotate(labDir, cache_dir):
    inputDataFiles = []

    for (dirpath, dirnames, filenames) in os.walk(labDir):
        for filename in filenames:
            # Check if the filename matches the regex and does not contain "Walking"
            if re.search("^(L|ID)[0-9]+.*.ontrackclassifier.joblib$", filename) and "Walk" not in filename:
                inputDataFiles.append(os.path.join(Path(dirpath), Path(filename)))


    cacheFileName = f"annoatatedLabData.joblib"
    annotatedLabData = os.path.join(cache_dir, cacheFileName)
    isCached_aLD = os.path.exists(annotatedLabData)

    # If cached, then jump directly to feature extraction
    if not isCached_aLD:
        print(f'[info@load_lab_data.chunk] -> Annotating lab data..')
        c3dProcessed = []
        num_None = 0 

        for i in tqdm(range(0, len(inputDataFiles)), desc="Loading lab data"):
            inputData = joblib.load(inputDataFiles[i])
            if inputData is not None:
                c3dProcessed.append(inputData)
            else:
                num_None += 1

        print(f"[info@load_lab_data.chunk] -> Number of Files with None data: {num_None}")

        '''
        Preprocess: segment data into 'Movement' and 'Others' with annotation label
        Estimated running time: 36 secs
        '''

        def annotateSegment(data, startIndex, endIndex, filename):
            return pd.DataFrame({
                "filename": filename,
                "accelX": data["accelX"][startIndex:endIndex],
                "accelY": data["accelY"][startIndex:endIndex],
                "accelZ": data["accelZ"][startIndex:endIndex],
                "accel": data["accel"][startIndex:endIndex],
            })

        def applyAnnotation(data, filename):
            startIndex = 0
            if "movement" not in data:
                return
            
            isMovement = data["movement"][0]
            for i in tqdm(range(1, len(data["movement"])), leave=False):
                if data["movement"][i] != isMovement:
                    segment = annotateSegment(data, startIndex, i-1, filename)
                    if isMovement == 1 and len(segment["accel"]) > 1000:
                        movementSegments.append(segment)
                    elif len(segment["accel"]) > 1000:
                        otherSegments.append(segment)
                    startIndex = i
                isMovement = data["movement"][i]
            
            # Add the last segment
            segment = annotateSegment(data, startIndex, len(data["movement"]), filename)
            if isMovement == 1 and len(segment["accel"]) > 1000:
                movementSegments.append(segment)
            elif len(segment["accel"]) > 1000:
                otherSegments.append(segment)

        movementSegments = []
        otherSegments = []

        for i in tqdm(range(0, len(c3dProcessed)), desc="Processing lab data"):
            applyAnnotation(c3dProcessed[i]["left"], c3dProcessed[i]["filename"] + str(" left"))
            applyAnnotation(c3dProcessed[i]["right"], c3dProcessed[i]["filename"] + str(" right"))

        # Save the annotated data
        joblib.dump({
            'movementSegments': movementSegments,
            'otherSegments': otherSegments
        }, annotatedLabData)
        print("[info@load_lab_data.chunk] -> Annotated lab data saved.")
    else:
        print("[info@load_lab_data.chunk] -> Annotated lab data loaded from cache.")
        data = joblib.load(annotatedLabData)
        movementSegments = data['movementSegments']
        otherSegments = data['otherSegments']

    return movementSegments, otherSegments

def chunk(movementSegments, otherSegments):
    '''
    Preprocess: chunk lab data with overlapping windows
    '''
    window_size = int(WINDOW_SIZE * LAB_SAMPLING_RATE)  # 1500 Hz
    step_size = int(window_size * OVERLAPPING_PERCENTAGE)

    print(f"[info@load_lab_data.chunk] -> window_size: {window_size}, step_size: {step_size}")

    cacheFileName = f"LAB_ws{WINDOW_SIZE}.joblib"
    checkpoint_ws = os.path.join(cache_dir, cacheFileName)
    isCached = os.path.exists(checkpoint_ws)

    if not isCached:
        print("[info@load_lab_data.chunk] -> Chunking examples...")
        chunkedMovement = []
        chunkedOther = []
        def chunkExample(example):
            chunks = []
            for i in range(0, len(example.index) - window_size + 1, step_size):  # Step size for overlap
                chunk = example.iloc[i:i + window_size]
                if len(chunk.index) == window_size:  # Ensure the chunk size is exactly 5000
                    chunks.append(chunk)
            return chunks

        def chunkAllExamples(examples):
            l = []
            for i in range(0, len(examples)):
                l.append(chunkExample(examples[i]))
            return list(chain.from_iterable(l))

        # Example chunking for Movement and Other datasets
        chunkedMovement = chunkAllExamples(movementSegments)
        chunkedOther = chunkAllExamples(otherSegments)
        # Save the checkpoint
        data = {
            'movement': chunkedMovement,
            'other': chunkedOther
        }
        joblib.dump(data, checkpoint_ws, compress=3)
        print("[info@load_lab_data.chunk] -> Checkpoint saved.")
    else:
        print("[info@load_lab_data.chunk] -> Loading checkpoint...")
        data = joblib.load(checkpoint_ws)
        chunkedMovement = data['movement']
        chunkedOther = data['other']
        print("[info@load_lab_data.chunk] -> Checkpoint loaded.")

    return chunkedMovement, chunkedOther

# Optimized resample function
def resample(d):
    if len(d) != (WINDOW_SIZE * LAB_SAMPLING_RATE):
        print(f"[Warning@load_lab_data] -> Chunk size {len(d)} does not match expected window size {WINDOW_SIZE}. Make sure the correct cache file is used.")
        return None
        
    filename = d.iloc[0].filename  # Extract filename
    # Set index and resample numeric columns only
    numeric_columns = ['accelX', 'accelY', 'accelZ', 'accel']
    # df = d[numeric_columns].set_index(pd.TimedeltaIndex([str(x / 5000) + "seconds" for x in range(window_size)], freq="infer"))
    df = d[numeric_columns].set_index(pd.TimedeltaIndex([str(x / LAB_SAMPLING_RATE) + "seconds" for x in range(len(d))], freq="infer"))
    df_resampled = df.resample("0.05s").mean().interpolate().dropna()  # Perform resampling and interpolation on numeric columns
    df_resampled["filename"] = filename  # Add the filename back to the dataframe
    return df_resampled

# Batch processing for better performance
def process_in_batches(chunks, batch_size=50):
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        results.extend(map(lambda x: resample(x), batch))
    gc.collect()  # Free memory after each batch
    return results

# Use joblib for parallel execution
def process_with_parallel(chunks):
    return Parallel(n_jobs=-1)(delayed(lambda x: resample(x))(chunk) for chunk in chunks)

def flatten(chunkedMovement, chunkedOther, method='parallel'):
    # print(f"[info@load_lab_data.flatten] -> Flattening chunked data...")
    flatMovementLab = process_with_parallel(chunkedMovement)
    flatOtherLab = process_with_parallel(chunkedOther)

    if method == 'batch':
        flatMovementLab = process_in_batches(chunkedMovement, batch_size=50)
        flatOtherLab = process_in_batches(chunkedOther, batch_size=50)
    else:
        # Delete the accel and filename columns
        flatMovementLab = [df.drop(["filename", "accel"], axis=1) for df in tqdm(flatMovementLab, desc="[info@load_lab_data.flatten] -> Flattening Movement Data", leave=False)]
        flatOtherLab = [df.drop(["filename", "accel"], axis=1) for df in tqdm(flatOtherLab, desc="[info@load_lab_data.flatten] -> Flattening Other Data", leave=False)]

    return flatMovementLab, flatOtherLab


def load_lab_data():
    # 1. Annotate the lab data
    movementSegments, otherSegments = annotate(lab_data_dir, cache_dir)

    # 2. Chunk the annotated data into overlapping windows
    chunkedMovement, chunkedOther = chunk(movementSegments, otherSegments)

    # 3. Flatten the chunked data into a format suitable for training
    flatMovementLab, flatOtherLab = flatten(chunkedMovement, chunkedOther, method='parallel')

    return flatMovementLab, flatOtherLab