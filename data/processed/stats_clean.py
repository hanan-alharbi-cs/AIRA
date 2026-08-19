import pandas as pd

def clean_stats():
    # Load raw stats
    df = pd.read_csv("data/raw/stats.csv")

    # Remove missing rows
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Ensure numeric columns are valid
    numeric_cols = ["goals", "assists", "yellow_cards", "red_cards", "matches_played"]
    for col in numeric_cols:
        df = df[df[col] >= 0]

    # Load players to validate player_id
    players = pd.read_csv("data/raw/players.csv")
    valid_ids = players["id"].unique()

    df = df[df["player_id"].isin(valid_ids)]

    # Save cleaned file
    df.to_csv("data/processed/stats_clean.csv", index=False)
    print("Cleaned stats_clean.csv")

if __name__=="__main__":
    clean_stats()