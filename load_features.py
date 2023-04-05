import pandas as pd
from typing import Generator


def load_openface_features(features_path: Generator) -> pd.DataFrame:
    """Loads the .csv files of OpenFace features as DataFrame objects.

    Iterates over the directory containing data samples. The features
    from each sample are loaded into a dictionary of dataframes.
    """
    samples = {}
    for path in features_path:
        str_path = str(path)
        start = str_path.index("\\")
        end = str_path.index(".")
        sample = str_path[start + 1 : end]

        sample_df = pd.read_csv(path)
        sample_df.columns = sample_df.columns.str.replace(" ", "")
        samples[sample] = sample_df
    return samples
