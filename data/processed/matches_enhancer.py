import pandas as pd
import numpy as np

def enhance_matches():
    df = pd.read_csv("data/processed/matches_clean.csv")

    # توليد minutes_played (بين 45 و 95 دقيقة)
    df["minutes_played"] = np.random.randint(45, 95, size=len(df))

    df.to_csv("data/processed/matches_clean.csv", index=False)
    print("matches_clean.csv تم تحديثه بالحقول المطلوبة")

if __name__ == "__main__":
    enhance_matches()
