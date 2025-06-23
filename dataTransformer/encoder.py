from sklearn.preprocessing import LabelEncoder
import os
import joblib
import sys
from loguru import logger

from config import cache_dir, encode_path

# Configure loguru logger
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

def encode(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    mapping = {str(k): v for k, v in zip(encoder.classes_, range(len(encoder.classes_)))}
    logger.info(f"Encoder: {mapping}")
    # Save the mapping encoder
    joblib.dump(encoder, encode_path)
    logger.info(f"Encoder saved to {encode_path}.")
    return encoded_labels

def get_encoder():
    """Get the saved encoder from the cache directory."""
    if os.path.exists(encode_path):
        return joblib.load(encode_path)
    else:
        raise FileNotFoundError(f"No encoder found at {encode_path}")