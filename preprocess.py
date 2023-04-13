"""Functions for performing any required preprocessing of data samples."""
from typing import Dict
import pandas as pd
import numpy as np

def hog_windowing(raw_samples: Dict[str, pd.DataFrame]): # do some type hints
    windowed_samples = np.array()
    for sample in raw_samples:
        windowed_samples = 