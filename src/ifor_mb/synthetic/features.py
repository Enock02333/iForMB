from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import SyntheticSchema

SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "WARN": 1, "ERROR": 2, "FATAL": 3, "CRITICAL": 3}


def add_temporal_and_numeric_features(frame: pd.DataFrame, schema: SyntheticSchema) -> pd.DataFrame:
    data = frame.copy()
    timestamp = pd.to_datetime(data[schema.timestamp], errors="coerce", utc=True)
    hour = timestamp.dt.hour.fillna(0).astype(float)
    weekday = timestamp.dt.dayofweek.fillna(0).astype(float)
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["duration_numeric"] = pd.to_numeric(data[schema.duration], errors="coerce").fillna(0.0)
    data["severity_ordinal"] = data[schema.severity].astype(str).str.upper().map(SEVERITY_ORDER).fillna(-1)
    data["message_length"] = data[schema.message].astype(str).str.len().astype(float)
    return data


def tabular_preprocessor(schema: SyntheticSchema) -> ColumnTransformer:
    numeric = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "duration_numeric", "severity_ordinal", "message_length"]
    categorical = [schema.service, schema.user, schema.client]
    return ColumnTransformer(
        [("numeric", StandardScaler(), numeric), ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="drop",
    )


def message_vectorizer(n_features: int = 256) -> HashingVectorizer:
    return HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2", lowercase=True)
