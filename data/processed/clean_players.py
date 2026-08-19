import pandas as pd

def clean_players():
    df = pd.read_csv("data/raw/players.csv")

    # إعادة تسمية العمود id إلى player_id
    df.rename(columns={"id": "player_id"}, inplace=True)

    # إزالة الصفوف الناقصة
    df = df.dropna()

    # إزالة التكرار
    df = df.drop_duplicates()

    # التأكد من الأعمار
    df = df[(df["age"] >= 16) & (df["age"] <= 45)]

    # التأكد من المراكز
    valid_positions = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    df = df[df["position"].isin(valid_positions)]

    df.to_csv("data/processed/players_clean.csv", index=False)
    print("cleaned players_clean.csv")

if __name__ == "__main__":
    clean_players()
