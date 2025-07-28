# The Pipeline of data mapping process

## 1. Calculate Rotation Matrix

```python
rotation_matrix_aw_to_vicon = np.array(
    [[ 0.58377147,  0.28237329,  0.76123334],
     [ 0.36897715,  0.74289845, -0.55853179],
     [-0.72323352,  0.60693263,  0.32949362]])
```

## 2. Build training set for mapping model

Need to concatenate more data together to build a larger training set.

### 2.1 Load data

### 2.2 Preprocess

- Downsampling
- Bandpass filter (remove low freq gravity and high freq noise)
- Normalization [Optional]

### 2.3 Apply the Rotation Matrix

### 2.4 Initial time alignment (manual or `initial_time_alignment`)

### 2.5 DTW

```python
alignment = dtw(x=series1, y=series2, 
                dist_method="euclidean",
                step_pattern="symmetric2")
```

## 3. Training the Mapping Model and Evaluation

```python
    vicon = pd.DataFrame(aligned_vicon_series, columns=['accelX','accelY','accelZ'])
    aw = pd.DataFrame(aligned_aw_series, columns=['accelX','accelY','accelZ'])
    mapping_model = train_mapping_model(vicon, aw)
    joblib.dump(mapping_model, model_save_path)
```

## 4. Apply the model to real data