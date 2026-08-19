import pandas as pd
import numpy as np

def generate_validation_report():
    report = []

    # قراءة الملفات
    players = pd.read_csv("data/processed/players_clean.csv")
    stats = pd.read_csv("data/processed/stats_clean.csv")
    matches = pd.read_csv("data/processed/player_matches.csv")
    baselines = pd.read_csv("data/processed/baselines.csv")
    metrics = pd.read_csv("data/processed/metrics.csv")

    # -----------------------------
    # 1) Missing Values Check
    # -----------------------------
    report.append("=== Missing Values Check ===")
    for name, df in [
        ("Players", players),
        ("Stats", stats),
        ("Matches", matches),
        ("Baselines", baselines),
        ("Metrics", metrics)
    ]:
        missing = df.isnull().sum()
        report.append(f"\n{name} Missing Values:\n{missing}")

    # -----------------------------
    # 2) Schema Check
    # -----------------------------
    report.append("\n=== Schema Check ===")
    expected_columns = {
        "players": ["player_id", "name", "age", "position", "team"],
        "stats": ["player_id", "goals", "assists", "yellow_cards", "red_cards",
                  "matches_played", "reaction_time_ms", "sleep_duration", "recovery_score"],
        "matches": ["player_id", "match_id", "minutes_played"],
        "baselines": ["player_id", "baseline_training_load", "baseline_reaction_time",
                      "baseline_sleep", "baseline_recovery"],
        "metrics": ["player_id", "session_load", "acute_load", "chronic_load", "acwr",
                    "sleep_deviation", "reaction_time_deviation", "recovery_drop", "warning_points"]
    }

    for name, df, cols in [
        ("Players", players, expected_columns["players"]),
        ("Stats", stats, expected_columns["stats"]),
        ("Matches", matches, expected_columns["matches"]),
        ("Baselines", baselines, expected_columns["baselines"]),
        ("Metrics", metrics, expected_columns["metrics"])
    ]:
        missing_cols = set(cols) - set(df.columns)
        report.append(f"\n{name} Missing Columns: {missing_cols}")

    # -----------------------------
    # 3) Outliers Check
    # -----------------------------
    report.append("\n=== Outliers Check ===")
    numeric_cols = metrics.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = metrics[col].quantile(0.25)
        q3 = metrics[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = metrics[(metrics[col] < lower) | (metrics[col] > upper)]
        report.append(f"\nOutliers in {col}: {len(outliers)}")

    # -----------------------------
    # 4) Baseline Sanity Check
    # -----------------------------
    report.append("\n=== Baseline Sanity Check ===")
    report.append(f"Baseline Sleep Range: {baselines['baseline_sleep'].min()} - {baselines['baseline_sleep'].max()}")
    report.append(f"Baseline Reaction Time Range: {baselines['baseline_reaction_time'].min()} - {baselines['baseline_reaction_time'].max()}")

    # -----------------------------
    # 5) Metrics Sanity Check
    # -----------------------------
    report.append("\n=== Metrics Sanity Check ===")
    report.append(f"ACWR Range: {metrics['acwr'].min()} - {metrics['acwr'].max()}")
    report.append(f"Warning Points Range: {metrics['warning_points'].min()} - {metrics['warning_points'].max()}")

    # -----------------------------
    # 6) Player-Match Linkage Check
    # -----------------------------
    report.append("\n=== Player-Match Linkage Check ===")
    unique_players = matches["player_id"].nunique()
    report.append(f"Players in player_matches.csv: {unique_players}")

    # حفظ التقرير
    with open("data/processed/validation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("validation_report.txt جاهز")

if __name__ == "__main__":
    generate_validation_report()
