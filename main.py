"""Main script."""
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import (
    datasets,
    layers,
    models,
)
import matplotlib.pyplot as plt
from load_features import load_openface_features, load_openface_hog


FEATURES_PATH = Path("openface_features")
HOG_FEATURES_PATH = Path("openface_features\\hog_features")

samples = load_openface_features(FEATURES_PATH)
sample_hog = load_openface_hog(HOG_FEATURES_PATH)

print("complete")
