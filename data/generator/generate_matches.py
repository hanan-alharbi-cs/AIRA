import random
import pandas as pd

def generate_matches(num_matches=30, random_seed=42):
    random.seed(random_seed)

    # Load players
    players = pd.read_csv("data/raw/players.csv")

    # Extract teams
    teams = players["team"].unique()

    matches = []

    for i in range(1, num_matches + 1):
        match_id = f"M{i:03d}"

        # Pick two different teams
        team_home, team_away = random.sample(list(teams), 2)

        # Random score
        home_score = random.randint(0, 5)
        away_score = random.randint(0, 5)

        matches.append({
            "match_id": match_id,
            "team_home": team_home,
            "team_away": team_away,
            "home_score": home_score,
            "away_score": away_score
        })

    return pd.DataFrame(matches)

if __name__ == "__main__":
    df = generate_matches()
    df.to_csv("data/raw/matches.csv", index=False)
    print("Generated matches.csv")
