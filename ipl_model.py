# ipl_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load data
df = pd.read_csv("IPL.csv",low_memory=False)

# ✅ Convert season column to numeric (force errors to NaN)
df["season"] = pd.to_numeric(df["season"], errors="coerce")

# ✅ Drop rows where season is NaN (non-numeric)
df = df.dropna(subset=["season"])

# ✅ Now safely filter by integer range
df = df[(df["season"] >= 2020) & (df["season"] <= 2025)]

# Only 1st and 2nd innings
df = df[df["innings"] <= 2]

# Drop missing values
df = df.dropna(subset=["batting_team", "bowling_team", "batter", "bowler"])

# Add score and cumulative score
df["runs"] = df["runs_batter"] + df["runs_extras"]
df["current_score"] = df.groupby(["match_id", "innings"])["runs"].cumsum()

# Cumulative wickets
df["wickets"] = df["bowler_wicket"].fillna(0).astype(int)
df["wickets"] = df.groupby(["match_id", "innings"])["wickets"].cumsum()

# ✅ Get only unique teams from 2020–2025
all_teams = pd.concat([df["batting_team"], df["bowling_team"]]).unique()
team_encoder = LabelEncoder()
team_encoder.fit(all_teams)

# Encode both teams using same encoder
df["batting_team_enc"] = team_encoder.transform(df["batting_team"])
df["bowling_team_enc"] = team_encoder.transform(df["bowling_team"])

# Features and target
X = df[["batting_team_enc", "bowling_team_enc", "over", "ball", "current_score", "wickets"]]
y = df.groupby(["match_id", "innings"])["team_runs"].transform("max")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y.fillna(0), test_size=0.2, random_state=42)

# Train and save model
model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "ipl_score_model.pkl")
joblib.dump(team_encoder, "team_encoder.pkl")

# Prediction utilities
def predict_final_score(input_data):
    model = joblib.load("ipl_score_model.pkl")
    return model.predict([input_data])[0]

def get_team_names():
    encoder = joblib.load("team_encoder.pkl")
    return list(encoder.classes_)

def encode_team(name):
    encoder = joblib.load("team_encoder.pkl")
    return encoder.transform([name])[0]
