from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List


SCALE_COLUMNS = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
                 'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
                 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
                 'MMSE', 'FunctionalAssessment', 'ADL']


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns and duplicates (mirrors notebook cleaning)."""
    df = df.copy()
    for c in ['PatientID', 'DoctorInCharge']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    df = df.drop_duplicates()
    return df


def scale_features(df: pd.DataFrame, columns: List[str] = SCALE_COLUMNS) -> pd.DataFrame:
    """Fit StandardScaler on the selected columns and return transformed df."""
    df = df.copy()
    scaler = StandardScaler()
    # Only scale columns that are present
    cols = [c for c in columns if c in df.columns]
    if cols:
        df[cols] = scaler.fit_transform(df[cols])
    return df
