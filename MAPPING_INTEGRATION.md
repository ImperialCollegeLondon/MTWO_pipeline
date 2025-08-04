# Data Mapping Integration

## Overview

This document describes the integration of the coordinate system mapping functionality into the main MTWO pipeline. The mapping system transforms data from Vicon Lab coordinate system to Apple Watch coordinate system using trained mapping models.

## Files Added/Modified

### New Files
- `dataTransformer/data_mapper.py` - Core mapping functionality
- `test_mapping.py` - Test script for mapping functionality

### Modified Files
- `main.py` - Integrated mapping into both `train_MTWO()` and `train_MO()` functions
- `config.py` - Added mapping configuration parameters

## Configuration

The mapping functionality is controlled by parameters in `config.py`:

```python
# ------------------Mapping Configuration------------------
ENABLE_MAPPING = True  # Enable/disable coordinate system mapping
MAPPING_ALIGNMENT_METHOD = 'none'  # Alignment method: 'none', 'rotation_matrix', 'procrustes'
MAPPING_MODEL_PATH = None  # Path to mapping model (None = auto-detect)
```

### Configuration Parameters

- **ENABLE_MAPPING**: Set to `True` to enable mapping, `False` to disable
- **MAPPING_ALIGNMENT_METHOD**: Specifies which trained model to use:
  - `'none'` - Uses the model trained without alignment
  - `'rotation_matrix'` - Uses the model trained with rotation matrix alignment
  - `'procrustes'` - Uses the model trained with Procrustes alignment
- **MAPPING_MODEL_PATH**: Explicit path to mapping model (optional, auto-detects if None)

## How It Works

### Pipeline Integration

The mapping is applied at **Step 1.5** in both training functions, immediately after data loading:

```
Step 1: Load Data
Step 1.5: Apply Mapping Transformation (if enabled)
Step 2: Data Augmentation
Step 3: Feature Extraction
...
```

### Data Flow

1. **Data Loading**: Original data is loaded from various sources
2. **Mapping Application**: If enabled, the mapping model transforms the data
3. **Continue Pipeline**: The transformed data proceeds through the normal pipeline

### Model Auto-Detection

The system automatically searches for mapping models in these locations:
1. Root directory: `mapping_model.joblib`
2. Alignment-specific directory: `mapping/mapping_models_{method}/mapping_model.joblib`
3. Fallback to 'none' method: `mapping/mapping_models_none/mapping_model.joblib`

## Usage Examples

### Basic Usage (Enabled by Default)

Simply run the main pipeline as usual:

```python
# In main.py
if __name__ == '__main__':
    train_MO()  # Mapping will be applied automatically
```

### Disable Mapping

```python
# In config.py
ENABLE_MAPPING = False
```

### Use Different Alignment Method

```python
# In config.py
MAPPING_ALIGNMENT_METHOD = 'rotation_matrix'
```

### Use Custom Model Path

```python
# In config.py
MAPPING_MODEL_PATH = r"path/to/your/custom/mapping_model.joblib"
```

## Data Format Support

The mapping system supports multiple data formats:

### DataFrame Lists
- Format: `List[pd.DataFrame]`
- Columns: `['accelerationX', 'accelerationY', 'accelerationZ']` or `['accelX', 'accelY', 'accelZ']`
- Used by: Custom dataset loading

### Numpy Arrays
- Format: `np.ndarray`
- Shapes: 
  - 2D: `(n_samples, 3)` 
  - 3D: `(n_windows, window_size, 3)`
- Used by: Windowed data and original sources

## Testing

Run the test script to verify mapping functionality:

```bash
python test_mapping.py
```

This will test:
- Model loading and auto-detection
- DataFrame transformation
- Array transformation (2D and 3D)
- List of DataFrames transformation
- Pass-through mode when no model is available

## Error Handling

The mapping system includes robust error handling:

- **No Model Found**: Operates in pass-through mode (data unchanged)
- **Transformation Errors**: Returns original data with warning
- **Invalid Data Format**: Logs error and returns original data

## Performance Considerations

- **Model Loading**: Models are loaded once and reused
- **Memory Usage**: Transformations create copies to preserve original data
- **Processing Time**: Mapping adds minimal overhead to the pipeline

## Troubleshooting

### Common Issues

1. **"No mapping model found"**
   - Check if `mapping_model.joblib` exists in the expected locations
   - Verify the alignment method is correct
   - Set `MAPPING_MODEL_PATH` to explicit path if needed

2. **"Failed to apply mapping transformation"**
   - Check if data has the correct column names
   - Verify data shape is compatible with the model
   - Check model file integrity

3. **Memory issues with large datasets**
   - Consider processing data in batches
   - Monitor memory usage during transformation

### Debug Mode

Enable debug logging to see detailed mapping information:

```python
from config import getLogger
logger = getLogger('DEBUG')
```

## Future Enhancements

Potential improvements for the mapping system:

1. **Batch Processing**: Process large datasets in chunks
2. **Model Validation**: Verify model compatibility before applying
3. **Performance Metrics**: Track transformation time and accuracy
4. **Multiple Model Support**: Use different models for different data types
5. **Real-time Mapping**: Apply mapping to streaming data
