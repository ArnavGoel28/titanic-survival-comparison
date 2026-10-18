"""Shared training-only preparation for the two Titanic notebooks."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent
NUMERIC = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]


def load_data():
    path = DATA_DIR / "input" / "titanic.csv"
    expected_hash = "7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, "Unexpected input version"
    return pd.read_csv(path)


def prepare_data():
    """Return X_train, X_test, y_train, y_test using the saved passenger split."""
    frame = load_data()
    split = json.loads((DATA_DIR / "input" / "split.json").read_text())
    training = frame[frame.PassengerId.isin(split["train_ids"])].copy()
    test = frame[frame.PassengerId.isin(split["holdout_ids"])].copy()
    assert set(training.PassengerId).isdisjoint(test.PassengerId)
    assert len(training) + len(test) == len(frame)

    X_train = training[NUMERIC + CATEGORICAL].copy()
    X_test = test[NUMERIC + CATEGORICAL].copy()
    y_train, y_test = training.Survived, test.Survived

    # Fill both partitions using values learned only from training passengers.
    medians = X_train[NUMERIC].median()
    X_train[NUMERIC] = X_train[NUMERIC].fillna(medians)
    X_test[NUMERIC] = X_test[NUMERIC].fillna(medians)
    for column in CATEGORICAL:
        mode = X_train[column].mode().iloc[0]
        X_train[column] = X_train[column].fillna(mode)
        X_test[column] = X_test[column].fillna(mode)
        categories = sorted(X_train[column].unique())
        X_train[column] = pd.Categorical(X_train[column], categories=categories)
        X_test[column] = pd.Categorical(X_test[column], categories=categories)

    X_train = pd.get_dummies(X_train, columns=CATEGORICAL, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL, drop_first=True, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    scaler = StandardScaler()
    X_train[NUMERIC] = scaler.fit_transform(X_train[NUMERIC])
    X_test[NUMERIC] = scaler.transform(X_test[NUMERIC])
    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_test.to_numpy()).all()
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print("Training records:", len(X_train))
    print("Test records:", len(X_test))
    print("Prepared columns:", X_train.columns.tolist())
    print("Missing prepared values:", X_train.isna().sum().sum(), X_test.isna().sum().sum())
