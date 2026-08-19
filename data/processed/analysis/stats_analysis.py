import pandas as pd

def analyze_stats():
    df = pd.read_csv("data/processed/stats_clean.csv")

    # أفضل 5 هدّافين
    top_scorers = df.sort_values(by="goals", ascending=False).head(5)

    # أفضل 5 صانعين لعب
    top_assists = df.sort_values(by="assists", ascending=False).head(5)

    # أكثر لاعبين يحصلون بطاقات صفراء
    top_yellow = df.sort_values(by="yellow_cards", ascending=False).head(5)

    # أكثر لاعبين يحصلون بطاقات حمراء
    top_red = df.sort_values(by="red_cards", ascending=False).head(5)

    print("\nTop 5 Goal Scorers:")
    print(top_scorers)

    print("\nTop 5 Assist Leaders:")
    print(top_assists)

    print("\nTop 5 Yellow Cards:")
    print(top_yellow)

    print("\nTop 5 Red Cards:")
    print(top_red)

if __name__=="__main__":
    analyze_stats()