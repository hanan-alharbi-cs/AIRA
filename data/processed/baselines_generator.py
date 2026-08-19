import pandas as pd
import numpy as np

def generate_baselines():
    # قراءة الملفات
    players = pd.read_csv("data/processed/players_clean.csv")
    stats = pd.read_csv("data/processed/stats_clean.csv")
    matches = pd.read_csv("data/processed/player_matches.csv")
    injuries = pd.read_csv("data/processed/injuries_clean.csv")

    # إنشاء DataFrame للـ baselines
    baselines = pd.DataFrame()
    baselines["player_id"] = players["player_id"]

    # baseline_training_load = متوسط minutes_played
    if "minutes_played" in matches.columns:
        baseline_training = matches.groupby("player_id")["minutes_played"].mean().reset_index()
        baseline_training.rename(columns={"minutes_played": "baseline_training_load"}, inplace=True)
        baselines = baselines.merge(baseline_training, on="player_id", how="left")
    else:
        baselines["baseline_training_load"] = 0

    # baseline_reaction_time = متوسط reaction_time_ms
    if "reaction_time_ms" in stats.columns:
        baseline_reaction = stats.groupby("player_id")["reaction_time_ms"].mean().reset_index()
        baseline_reaction.rename(columns={"reaction_time_ms": "baseline_reaction_time"}, inplace=True)
        baselines = baselines.merge(baseline_reaction, on="player_id", how="left")
    else:
        baselines["baseline_reaction_time"] = 0

    # baseline_sleep = متوسط sleep_duration
    if "sleep_duration" in stats.columns:
        baseline_sleep = stats.groupby("player_id")["sleep_duration"].mean().reset_index()
        baseline_sleep.rename(columns={"sleep_duration": "baseline_sleep"}, inplace=True)
        baselines = baselines.merge(baseline_sleep, on="player_id", how="left")
    else:
        baselines["baseline_sleep"] = 0

    # baseline_recovery = متوسط recovery_score
    if "recovery_score" in stats.columns:
        baseline_recovery = stats.groupby("player_id")["recovery_score"].mean().reset_index()
        baseline_recovery.rename(columns={"recovery_score": "baseline_recovery"}, inplace=True)
        baselines = baselines.merge(baseline_recovery, on="player_id", how="left")
    else:
        baselines["baseline_recovery"] = 0

    # حفظ الملف النهائي
    baselines.to_csv("data/processed/baselines.csv", index=False)
    print("baselines.csv جاهز")

if __name__ == "__main__":
    generate_baselines()
