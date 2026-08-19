import pandas as pd

def clean_matches():
    # Load raw matches
    df = pd.read_csv("data/raw/matches.csv")

    # Remove missing rows
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Ensure teams are different
    df = df[df["team_home"] != df["team_away"]]

    # Ensure scores are valid numbers
    df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]

    # Load players to validate teams
    players = pd.read_csv("data/raw/players.csv")
    valid_teams = players["team"].unique()

    # Keep only matches with valid teams
    df = df[df["team_home"].isin(valid_teams)]
    df = df[df["team_away"].isin(valid_teams)]

    # Save cleaned file
    df.to_csv("data/processed/matches_clean.csv", index=False)
    print("Cleaned matches_clean.csv")

if __name__=="__main__":
    clean_matches()