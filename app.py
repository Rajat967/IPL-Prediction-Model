# app.py

import streamlit as st
from ipl_model import predict_final_score, get_team_names, encode_team

st.set_page_config(page_title="🏏 IPL Score Predictor", layout="centered")
st.title("🏏 IPL Score Prediction System")

st.markdown("""
This app predicts the **final score** of an IPL match and provides a mock **winning probability** based on current match context.

✅ Includes only teams from **IPL 2020 to 2025**  
❌ Player of the match prediction removed for simplicity
""")

# Get valid teams
teams = get_team_names()

# Sidebar inputs
st.sidebar.header("📋 Match Situation")

batting_team = st.sidebar.selectbox("Batting Team", teams)
bowling_team = st.sidebar.selectbox(
    "Bowling Team", [team for team in teams if team != batting_team]
)

over = st.sidebar.slider("Current Over", 0, 19, 5)
ball = st.sidebar.slider("Ball in Over", 1, 6, 3)
current_score = st.sidebar.number_input("Current Runs", min_value=0, value=50)
wickets = st.sidebar.slider("Wickets Fallen", 0, 10, 3)

# Predict
if st.sidebar.button("🎯 Predict Final Score"):
    try:
        batting_encoded = encode_team(batting_team)
        bowling_encoded = encode_team(bowling_team)
        input_features = [batting_encoded, bowling_encoded, over, ball, current_score, wickets]

        predicted_score = predict_final_score(input_features)
        st.success(f"🏏 Predicted Final Score: **{int(predicted_score)} runs**")

        win_prob = min(100, round((current_score / ((over * 6 + ball + 1)) * 1.5 + (10 - wickets) * 5), 2))
        st.info(f"📊 Winning Probability (approx.): **{win_prob}%**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made by Rajat using Streamlit + Random Forest")
