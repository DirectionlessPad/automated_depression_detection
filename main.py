from load_features import load_openface_features
from pathlib import Path

FEATURES_PATH = Path("openface_features").rglob("*.csv")

samples = load_openface_features(FEATURES_PATH)
