"""Functions for performing any required preprocessing of data samples."""
from typing import Dict, List, Any
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
    samples_dict: Dict[str, Dict[str, npt.NDArray]],
    labels_dict: Dict[str, Dict[str, Dict[str, int]]],
) -> Dict[str, Dict[str, List[Any]]]:
    """Takes the loaded raw .mat file data and windows it into 15 frame
    sections to be used as the input for the model.
    """
    windowed_samples = {
        "dev": [],
        "train": [],
        "test": [],
    }  # type: Dict[str, List[npt.NDArray]]
    windowed_samples_phqbinary_labels = {
        "dev": [],
        "train": [],
        "test": [],
    }  # type: Dict[str, List[int]]
    windowed_samples_phqscore_labels = {
        "dev": [],
        "train": [],
        "test": [],
    }  # type: Dict[str, List[int]]
    for dataset_split, samples in samples_dict.items():
        for participant_id, np_values in samples.items():
            for i in range(WINDOW_SIZE, np_values.shape[0], WINDOW_SIZE):
                windowed_samples[dataset_split].append(np_values[(i - WINDOW_SIZE) : i])
                windowed_samples_phqbinary_labels[dataset_split].append(
                    labels_dict[dataset_split][participant_id]["PHQ_Binary"]
                )
                windowed_samples_phqscore_labels[dataset_split].append(
                    labels_dict[dataset_split][participant_id]["PHQ_Score"]
                )
    return {
        "windowed samples": windowed_samples,
        "PHQ binary": windowed_samples_phqbinary_labels,
        "PHQ Score": windowed_samples_phqscore_labels,
    }
