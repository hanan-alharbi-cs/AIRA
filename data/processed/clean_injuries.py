import pandas as pd

def clean_injuries():
    # Load raw injuries
    df = pd.read_csv("data/raw/injuries.csv")

    # Remove missing rows
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Ensure recovery_days is valid
    df = df[df["recovery_days"] >= 0]

    # Load players to validate player_id
    players = pd.read_csv("data/raw/players.csv")
    valid_ids = players["id"].unique()

    df = df[df["player_id"].isin(valid_ids)]

    # Save cleaned file
    df.to_csv("data/processed/injuries_clean.csv", index=False)
    print("Cleaned injuries_clean.csv")

if __name__=="__main__":
    clean_injuries()