import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def build_model():
    df = pd.read_csv("modeling/training_data.csv")

    features = ["age", "goals", "assists", "yellow_cards", "red_cards", "matches_played"]
    X = df[features]
    y = df["injured"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

if __name__=="__main__":
    build_model()