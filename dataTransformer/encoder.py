from sklearn.preprocessing import LabelEncoder
import os
import joblib

from config import cache_dir, encode_path

def encode(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    mapping = {str(k): v for k, v in zip(encoder.classes_, range(len(encoder.classes_)))}
    print(f"[info@encoder] -> Encoder: {mapping}")
    # Save the mapping encoder
    joblib.dump(encoder, encode_path)
    print(f"[info@encoder] -> Encoder saved to {encode_path}.")
    return encoded_labels

def get_encoder():
    """Get the saved encoder from the cache directory."""
    if os.path.exists(encode_path):
        return joblib.load(encode_path)
    else:
        raise FileNotFoundError(f"No encoder found at {encode_path}")