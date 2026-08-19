import random
import pandas as pd

def generate_injuries(random_seed=42):
    random.seed(random_seed)

    # Load players
    players = pd.read_csv("data/raw/players.csv")

    injuries = []

    injury_types = [
        "Hamstring Strain",
        "Ankle Sprain",
        "Knee Ligament",
        "Shoulder Injury",
        "Back Pain",
        "Groin Pull",
        "No Injury"
    ]

    for _, row in players.iterrows():
        player_id = row["id"]

        # Random injury chance
        has_injury = random.choice([True, False, False])  # أقل احتمال للإصابة

        if has_injury:
            injury = random.choice(injury_types[:-1])  # بدون "No Injury"
            recovery_days = random.randint(3, 60)
        else:
            injury = "No Injury"
            recovery_days = 0

        injuries.append({
            "player_id": player_id,
            "injury": injury,
            "recovery_days": recovery_days
        })

    return pd.DataFrame(injuries)

if __name__ =="__main__":
    df = generate_injuries()
    df.to_csv("data/raw/injuries.csv", index=False)
    print("Generated injuries.csv")