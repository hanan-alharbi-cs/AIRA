import pandas as pd
import os

def prepare_data():
    # قراءة الملفات
    players = pd.read_csv("data/processed/players_clean.csv")
    stats = pd.read_csv("data/processed/stats_clean.csv")
    injuries = pd.read_csv("data/processed/injuries_clean.csv")

    # تنظيف الأعمدة
    def clean(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
        return df

    players = clean(players)
    stats = clean(stats)
    injuries = clean(injuries)

    # طباعة الأعمدة لمعرفة المشكلة
    print("Players columns:", players.columns)
    print("Stats columns:", stats.columns)
    print("Injuries columns:", injuries.columns)

    # التأكد من وجود player_id
    for df_name, df in [("players", players), ("stats", stats), ("injuries", injuries)]:
        if "player_id" not in df.columns:
            raise Exception(f"❌ ملف {df_name} لا يحتوي على عمود player_id بعد التنظيف")

    # دمج البيانات
    merged = players.merge(stats, on="player_id", how="left")

    # إضافة عمود الإصابة
    merged["injured"] = merged["player_id"].isin(injuries["player_id"]).astype(int)

    # إنشاء المجلد إذا لم يكن موجوداً
    output_dir = "data/processed/analysis/modeling"
    os.makedirs(output_dir, exist_ok=True)

    # حفظ الملف النهائي
    merged.to_csv(f"{output_dir}/training_data.csv", index=False)
    print("training_data.csv جاهز")

if __name__ == "__main__":
    prepare_data()
