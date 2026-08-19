"""
Train and evaluate the first temporal injury-risk ML baseline.

Important:
- Target = injury within the next 1–3 matches.
- Actual injury-match rows are excluded.
- Splitting is grouped by player_id to prevent leakage.
- This is synthetic development data, not clinical validation.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "temporal_injury_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ai"
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "temporal_injury_risk_baseline.joblib"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "temporal_baseline_predictions.csv"
)


FEATURES = [
    "session_load",
    "acute_load",
    "chronic_load",
    "acwr",
    "sleep_deviation",
    "reaction_time_deviation",
    "recovery_drop",
    "warning_points",
]

TARGET = "injury_soon_target"
GROUP = "player_id"


def load_dataset() -> pd.DataFrame:
    """Load and validate the temporal dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Temporal dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        GROUP,
        TARGET,
        *FEATURES,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Temporal dataset is missing columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Temporal dataset is empty."
        )

    if df[FEATURES].isna().any().any():
        raise ValueError(
            "Missing values found in model features."
        )

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Target must contain exactly two classes."
        )

    if df[GROUP].nunique() < 2:
        raise ValueError(
            "At least two players are required."
        )

    return df


def build_model() -> Pipeline:
    """Build an interpretable logistic-regression baseline."""

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                FEATURES,
            )
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocess,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def main() -> None:
    """Train and evaluate the grouped temporal baseline."""

    df = load_dataset()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    groups = df[GROUP].astype(str)

    print(
        f"Dataset shape: {df.shape}"
    )

    print()
    print("Target distribution:")
    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        f"Players: {groups.nunique()}"
    )

    class_counts = y.value_counts()

    minimum_class_count = int(
        class_counts.min()
    )

    if minimum_class_count < 2:
        raise ValueError(
            "Too few examples in one of the target classes."
        )

    # Keep every player's rows together.
    n_splits = min(
        5,
        minimum_class_count,
    )

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    model = build_model()

    probabilities = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    print()
    print("Grouped cross-validation results")
    print("--------------------------------")
    print(
        f"PR-AUC:    {pr_auc:.3f}"
    )
    print(
        f"ROC-AUC:   {roc_auc:.3f}"
    )
    print(
        f"Precision: {precision:.3f}"
    )
    print(
        f"Recall:    {recall:.3f}"
    )
    print(
        f"F1:        {f1:.3f}"
    )

    print()
    print("Classification report")
    print("---------------------")
    print(
        classification_report(
            y,
            predictions,
            digits=3,
            zero_division=0,
        )
    )

    print("Confusion matrix")
    print("----------------")
    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    # Save out-of-fold predictions.
    prediction_output = df[
        [
            GROUP,
            "match_id",
            "match_number",
            "matches_until_injury",
            TARGET,
        ]
    ].copy()

    prediction_output[
        "predicted_probability"
    ] = probabilities

    prediction_output[
        "predicted_class"
    ] = predictions

    PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # Fit final integration model on all data.
    model.fit(
        X,
        y,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print()
    print(
        f"Saved predictions: {PREDICTIONS_FILE}"
    )

    print(
        f"Saved model: {MODEL_FILE}"
    )

    print()
    print(
        "NOTE: This model uses synthetic development data."
    )


if __name__ == "__main__":
    main()