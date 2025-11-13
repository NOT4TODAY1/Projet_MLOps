import pandas as pd
from typing import Tuple


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(csv_path)
    return df


def split_features_target(df: pd.DataFrame, target_col: str = 'Diagnosis') -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
