"""Main script."""
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import (
    datasets,
    layers,
    models,
)
import matplotlib.pyplot as plt
from load_features import load_daic_openface_features, load_openface_hog
from load_labels import load_daic_labels
from preprocess import hog_windowing


FEATURES_PATH = Path("openface_features")
DAIC_LABELS = Path("daic_labels")
HOG_FEATURES_PATH = Path("openface_features\\hog_features")

# samples = load_daic_openface_features(FEATURES_PATH)
# sample_hog = load_openface_hog(HOG_FEATURES_PATH)

labels = load_daic_labels(DAIC_LABELS)

# windowed_hog_samples = hog_windowing(sample_hog)
# call a function to preprocess the features (need to write the function, possibly in a
# different module) this function should take in the 'sample_hog' object and labels and
# return two tensors, one with the HOG features for each window of the sample, and one
# with the corresponding labels.

# model = models.Sequential()

print("complete")
