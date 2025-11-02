"""Shared utilities for training scripts in the `train/` folder.

Functions:
 - load_dataset: loads dataset and builds target
 - get_features_and_target: returns X, y
 - make_numeric_pipeline: imputer + scaler for numeric features
 - build_preprocessor: ColumnTransformer that applies numeric pipeline

This file centralizes data loading and preprocessing so each model file
remains small and runnable on its own.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
BACKEND_DIR = PROJECT_ROOT / 'backend'


def load_dataset() -> pd.DataFrame:
    """Load dataset from data folder and create a synthetic target.

    Raises FileNotFoundError when the CSV can't be found.
    """
    csv_path = DATA_DIR / 'studataset.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    # same synthetic target as original train script
    df['depression'] = ((df['failures'] > 0) | (df['absences'] > 15) | (df['G3'] < 8)).astype(int)
    return df


def get_features_and_target(df):
    """Return (X, y) using the feature list from the original script."""
    features = ['failures', 'absences', 'goout', 'Dalc', 'Walc', 'studytime', 'health', 'G3']
    target = 'depression'
    X = df[features]
    y = df[target]
    return X, y


def make_numeric_pipeline() -> Pipeline:
    """Simple numeric preprocessing pipeline (median impute + scaling)."""
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer that applies numeric pipeline to all feature columns."""
    # there are 8 numeric features in the features list above
    return ColumnTransformer(transformers=[('num', make_numeric_pipeline(), list(range(8)))])


def update_metrics(metric_key: str, value: float):
    """Update (create or merge) the `backend/metrics.json` file with one key."""
    metrics_path = BACKEND_DIR / 'metrics.json'
    metrics = {}
    try:
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    except Exception:
        # if reading/parsing fails, overwrite with a fresh dict
        metrics = {}

    metrics[metric_key] = float(value)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
