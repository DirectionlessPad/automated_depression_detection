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
        start = str_path.rindex("\\")
        end = str_path.rindex(".")
        sample = str_path[start + 1 : end]
        # Read the csv file and store as dataframe.
        sample_df = pd.read_csv(path)
        sample_df.columns = sample_df.columns.str.replace(" ", "")
        samples[sample] = sample_df
    return samples


def load_openface_hog(features_path: Generator) -> pd.DataFrame:
    samples = {}
    for path in features_path:
        str_path = str(path)
        start = str_path.rindex("\\")
        end = str_path.rindex("H")
        sample = str_path[start + 1 : end]
        # Read the csv file and store as dataframe.
        sample_df = pd.read_csv(path, header=None)
        breakpoint()
        samples[sample] = sample_df
    return samples
