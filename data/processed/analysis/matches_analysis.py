import pandas as pd

def analyze_matches():
    df = pd.read_csv("data/processed/matches_clean.csv")

    # متوسط أهداف الفريق المضيف
    avg_home_score = df["home_score"].mean()

    # متوسط أهداف الفريق الضيف
    avg_away_score = df["away_score"].mean()

    # أعلى نتيجة مباراة
    highest_match = df.loc[(df["home_score"] + df["away_score"]).idxmax()]

    print("\nAverage Home Score:", avg_home_score)
    print("Average Away Score:", avg_away_score)

    print("\nHighest Scoring Match:")
    print(highest_match)

if __name__=="__main__":
    analyze_matches()