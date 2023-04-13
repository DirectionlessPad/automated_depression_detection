"""Functions for performing any required preprocessing of data samples."""
from typing import Dict
import pandas as pd
import numpy as np

FPS = 30
WINDOW_SIZE = int(FPS / 2)


def hog_windowing(df_samples: Dict[str, pd.DataFrame]) -> np.ndarray:
    """TODO"""
    windowed_samples = []
    for sample in df_samples:
        np_sample = df_samples[sample].values
        for i in range(WINDOW_SIZE, np_sample.shape[0], WINDOW_SIZE):
            windowed_samples.append(np_sample[(i - WINDOW_SIZE) : i])
            # also need to have a 'labels' object with the depression
            # rating for each window
    return np.array(windowed_samples)
