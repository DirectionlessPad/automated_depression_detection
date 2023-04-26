"""Functions for loading data labels."""
from pathlib import Path
from typing import Dict
import pandas as pd


def load_daic_labels(label_path: Path) -> Dict[str, Dict[str, int]]:
    """TODO"""
    if not Path.exists(label_path):
        print("Directory does not exist. Check input feature directory.")
    loaded_labels: Dict[str, Dict] = {"dev": {}, "train": {}, "test": {}}
    paths = {
        "dev": label_path / "dev_split.csv",
        "train": label_path / "train_split.csv",
        "test": label_path / "test_split.csv",
    }
    for dataset_split, path in paths.items():
        split_df = pd.read_csv(path)
        split_dict = split_df.to_dict()
        for i in range(len(split_dict["Participant_ID"])):
            participant = str(split_dict["Participant_ID"][i])
            loaded_labels[dataset_split][participant] = {
                "PHQ_Binary": split_dict["PHQ_Binary"][i],
                "PHQ_Score": split_dict["PHQ_Binary"][i],
            }
    return loaded_labels
