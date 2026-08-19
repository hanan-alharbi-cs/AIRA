import random
import pandas as pd

def generate_stats(random_seed=42):
    random.seed(random_seed)

    # Load players
    players = pd.read_csv("data/raw/players.csv")

    stats = []

    for _, row in players.iterrows():
        player_id = row["id"]

        goals = random.randint(0, 20)
        assists = random.randint(0, 15)
        yellow_cards = random.randint(0, 5)
        red_cards = random.randint(0, 2)
        matches_played = random.randint(5, 30)

        stats.append({
            "player_id": player_id,
            "goals": goals,
            "assists": assists,
            "yellow_cards": yellow_cards,
            "red_cards": red_cards,
            "matches_played": matches_played
        })

    return pd.DataFrame(stats)

if __name__== "__main__":
    df = generate_stats()
    df.to_csv("data/raw/stats.csv", index=False)
    print("Generated stats.csv")