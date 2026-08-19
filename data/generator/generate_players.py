import random
import pandas as pd

def generate_players(num_players=20, random_seed=42):
    random.seed(random_seed)

    positions = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    teams = ["Team A", "Team B", "Team C", "Team D"]

    players = []
    for i in range(1, num_players + 1):
        player_id = f"P{i:03d}"
        name = f"Player {i}"
        age = random.randint(18, 35)
        position = random.choice(positions)
        team = random.choice(teams)

        players.append({
            "id": player_id,
            "name": name,
            "age": age,
            "position": position,
            "team": team
        })

    return pd.DataFrame(players)

if __name__ == "__main__":
    df = generate_players()
    df.to_csv("data/raw/players.csv", index=False)
    print("Generated players.csv")