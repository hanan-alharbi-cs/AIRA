"""
Train and evaluate the first AthleteGuard ML baseline.

Important:
- The target is player-level injury association.
- Only pre-injury-style engineered features are used.
- recovery_days and injury text are never used as model features.
- Because the dataset contains only 20 players, this is a baseline/PoC,
  not a clinically validated injury-prediction model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "injury_player_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "models"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "injury_risk_baseline.joblib"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "baseline_predictions.csv"
)


FEATURES = [
    "session_load_mean",
    "session_load_max",
    "acute_load_mean",
    "acute_load_max",
    "chronic_load_mean",
    "acwr_mean",
    "acwr_max",
    "sleep_deviation_mean",
    "sleep_deviation_min",
    "reaction_time_deviation_mean",
    "reaction_time_deviation_max",
    "recovery_drop_mean",
    "recovery_drop_min",
    "warning_points_mean",
    "warning_points_max",
]


def load_dataset() -> pd.DataFrame:
    """Load and validate the player-level dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "player_id",
        "injury_target",
        *FEATURES,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["player_id"].duplicated().any():
        raise ValueError(
            "Player-level dataset must contain one row per player."
        )

    if df[FEATURES].isna().any().any():
        raise ValueError(
            "Missing values found in model features."
        )

    return df


def build_model() -> Pipeline:
    """Create the first interpretable ML baseline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    """Train, evaluate, and save the baseline model."""

    df = load_dataset()

    X = df[FEATURES]
    y = df["injury_target"].astype(int)

    print("Dataset shape:", df.shape)
    print()
    print("Target distribution:")
    print(y.value_counts().sort_index().to_string())
    print()

    class_counts = y.value_counts()

    if len(class_counts) != 2:
        raise ValueError(
            "Binary training requires both target classes."
        )

    minimum_class_count = int(class_counts.min())

    if minimum_class_count < 2:
        raise ValueError(
            "At least two samples are required in each class."
        )

    # With only 9 positive and 11 negative players,
    # use a small stratified CV instead of a single arbitrary split.
    n_splits = min(
        5,
        minimum_class_count,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    model = build_model()

    # Out-of-fold predictions provide a more honest baseline
    # than evaluating on the same samples used for fitting.
    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    print("Cross-validation results")
    print("------------------------")
    print(f"Accuracy:          {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print(f"ROC-AUC:           {roc_auc:.3f}")
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
    print()

    prediction_df = pd.DataFrame(
        {
            "player_id": df["player_id"],
            "injury_target": y,
            "predicted_probability": probabilities,
            "predicted_class": predictions,
        }
    )

    PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # Fit the final baseline on all available player-level data
    # only after cross-validation has been completed.
    model.fit(X, y)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(f"Saved predictions: {PREDICTIONS_FILE}")
    print(f"Saved model:       {MODEL_FILE}")


if __name__ == "__main__":
    main()