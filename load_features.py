"""Functions for loading input features."""
from pathlib import Path
from typing import Dict
import pandas as pd
import scipy.io
import numpy.typing as npt

# Would probably be better to initialise a class instance for each of dev, train and test
# to house the data samples and manipulate. These can all be from the same class or classes
# that inherit from the same class

# These two function can almost certainly be made into one by either importing both together
# into a single objector by creating a single (more generic) function that can handle both.


# def load_openface_features(
#     features_path: Path,
# ) -> Dict[str, pd.DataFrame]:  # won't work for DAIC
#     """Loads the .csv files of OpenFace features as DataFrame objects.

#     Iterates over the directory containing data samples. The features
#     from each sample are loaded into a dictionary of dataframes.

#     Dictionary keys are the names of samples.
#     """
#     if not Path.exists(features_path):
#         print("Directory does not exist. Check input feature directory.")
#     path_generator = features_path.rglob("*.csv")
#     samples = {}
#     for path in path_generator:
#         str_path = str(path)
#         # The following may or may not work depending on the naming conventions of the samples.
#         start = str_path.rindex("\\")
#         end = str_path.rindex(".")
#         sample = str_path[start + 1 : end]
#         # Read the csv file and store as dataframe.
#         sample_df = pd.read_csv(path)
#         sample_df.columns = sample_df.columns.str.replace(" ", "")
#         samples[sample] = sample_df
#     if not samples:
#         print(
#             "No samples loaded, check the samples are available in the input directory."
#         )
#     return samples


def load_daic_openface_features(
    features_path: Path,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """TODO"""
    if not Path.exists(features_path):
        print("Directory does not exist. Check input feature directory.")
    loaded_features: Dict[str, Dict] = {"dev": {}, "train": {}, "test": {}}
    dev_path_generator = (features_path / "dev").glob("*")
    train_path_generator = (features_path / "train").glob("*")
    test_path_generator = (features_path / "test").glob("*")
    generators = {
        "dev": dev_path_generator,
        "train": train_path_generator,
        "test": test_path_generator,
    }
    for dataset_split, subset_dict in loaded_features.items():
        gen = generators[dataset_split]
        for path in gen:
            str_path = str(path)
            start = str_path.rindex("\\")
            end = str_path.rindex("_")
            participant_id = str_path[start + 1 : end]
            full_path = path / (
                "features/" + participant_id + "_OpenFace2.1.0_Pose_gaze_AUs.csv"
            )
            participant_id_df = pd.read_csv(full_path)
            participant_id_df.columns = participant_id_df.columns.str.replace(" ", "")
            subset_dict[participant_id] = participant_id_df
            # loaded_features[dataset_split][participant_id] = feature
        if not subset_dict:
            print(
                "No samples loaded, check the samples are available in the input directory."
            )
    return loaded_features


def load_openface_hog(
    features_path: Path,
) -> Dict[str, pd.DataFrame]:  # won't work for DAIC
    """Loads the .csv files of HOG features generated by OpenFace.

    Iterates over the directory containing data samples. The features
    from each sample are loaded into a dictionary of dataframes.

    Dictionary keys are the names of samples.
    """
    if not Path.exists(features_path):
        print("Directory does not exist. Check input feature directory.")
    path_generator = features_path.rglob("*.csv")
    samples = {}
    for path in path_generator:
        str_path = str(path)
        # The following may or may not work depending on the naming conventions of the samples.
        start = str_path.rindex("\\")
        end = str_path.rindex("H")
        participant_id = str_path[start + 1 : end]
        # Read the csv file and store as dataframe.
        samples_df = pd.read_csv(path, header=None)
        samples[participant_id] = samples_df
    if not samples:
        print(
            "No samples loaded, check the samples are available in the input directory."
        )
    return samples


def load_daic_resnet50_features(
    features_path: Path,
) -> Dict[str, Dict[str, npt.NDArray]]:
    """TODO"""
    if not Path.exists(features_path):
        print("Directory does not exist. Check input feature directory.")
    loaded_features = {"dev": {}, "train": {}, "test": {}}  # type: Dict[str, Dict]
    dev_path_generator = (features_path / "dev").glob("*")
    train_path_generator = (features_path / "train").glob("*")
    test_path_generator = (features_path / "test").glob("*")
    generators = {
        "dev": dev_path_generator,
        "train": train_path_generator,
        "test": test_path_generator,
    }
    for dataset_split, subset_dict in loaded_features.items():
        gen = generators[dataset_split]
        for path in gen:
            str_path = str(path)
            start = str_path.rindex("\\")
            end = str_path.rindex("_")
            participant_id = str_path[start + 1 : end]
            full_path = path / ("features/" + participant_id + "_CNN_ResNet.mat")
            matlab_file = scipy.io.loadmat(full_path)
            feature = matlab_file["feature"]
            subset_dict[participant_id] = feature
            # loaded_features[dataset_split][participant_id] = feature
        if not subset_dict:
            print(
                "No samples loaded, check the samples are available in the input directory."
            )
    return loaded_features
