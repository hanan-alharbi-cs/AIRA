import pandas as pd

def analyze_players():
    df = pd.read_csv("data/processed/players_clean.csv")

    # متوسط الأعمار
    avg_age = df["age"].mean()

    # أكثر مركز انتشار
    most_common_position = df["position"].value_counts().idxmax()

    # توزيع الفرق
    team_distribution = df["team"].value_counts()

    print("Average Age:", avg_age)
    print("Most Common Position:", most_common_position)
    print("\nTeam Distribution:")
    print(team_distribution)

if __name__=="__main__":
    analyze_players()