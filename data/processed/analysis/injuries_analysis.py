import pandas as pd

def analyze_injuries():
    df = pd.read_csv("data/processed/injuries_clean.csv")

    # عدد كل نوع إصابة
    injury_counts = df["injury"].value_counts()

    # متوسط أيام التعافي
    avg_recovery = df[df["recovery_days"] > 0]["recovery_days"].mean()

    print("\nInjury Types Count:")
    print(injury_counts)

    print("\nAverage Recovery Days:", avg_recovery)

if __name__=="__main__":
    analyze_injuries()