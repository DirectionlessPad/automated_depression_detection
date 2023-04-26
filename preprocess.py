"""Functions for performing any required preprocessing of data samples."""
from typing import Dict, List
import pandas as pd
import numpy as np
import numpy.typing as npt

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


def daic_resnet50_windowing(
    samples_dict: Dict[str, Dict[str, npt.NDArray]]
) -> Dict[str, Dict[str, List[npt.NDArray]]]:
    """Takes the loaded raw .mat file data and windows it into 15 frame
    sections to be used as the input for the model.
    """
    windowed_samples = {"dev": [], "train": [], "test": []}  # type: Dict[str, List]
    windowed_samples_labels = {
        "dev": [],
        "train": [],
        "test": [],
    }  # type: Dict[str, List]
    for dataset_split, samples in samples_dict.items():
        for participant_id, np_values in samples.items():
            for i in range(WINDOW_SIZE, np_values.shape[0], WINDOW_SIZE):
                windowed_samples[dataset_split].append(np_values[(i - WINDOW_SIZE) : i])
