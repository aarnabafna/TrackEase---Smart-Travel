import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import google.generativeai as genai

# --- API Keys ---
genai.configure(api_key="AIzaSyAMty2R33y3UHeC7jVLAll4YM_GxNK0gnc")  # Replace with your Gemini API key
WEATHER_API_KEY = "1afdd88fb14c4b25e2e9192d7eabbba9"  # Replace with your OpenWeatherMap API key
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# --- Load Data ---
df = pd.read_csv("india (1).csv")

df['Weather_Impact'] = 1.0
df['Adjusted_Time'] = df['Time_Minutes'] * df['Weather_Impact']
X = df[['Distance_km', 'Traffic', 'Base_Time', 'Weather_Impact']]
y = df['Adjusted_Time']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "final.pkl")

# --- Functions ---

def get_weather_impact(city):
    params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        wind_speed = data["wind"]["speed"]
        rain = data.get("rain", {}).get("1h", 0)

        weather_impact = 1 + ((temp - 25) * 0.002) + (wind_speed * 0.005) + (rain * 0.01)
        return max(0.8, min(1.2, weather_impact)), temp, wind_speed, rain

    return 1.0, None, None, None

def get_ai_recommendations(temp, wind_speed, rain):
    prompt = (
        f"Given the following weather conditions:\n"
        f"- Temperature: {temp}°C\n"
        f"- Wind Speed: {wind_speed} km/h\n"
        f"- Rainfall: {rain} mm\n\n"
        f"Give 3 short and friendly travel safety tips. Use emojis to make it engaging."
    )

    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        tips = [tip.strip("•-–1234567890. ") for tip in response.text.strip().split("\n") if tip.strip()]
        return tips[:3]  # Return only the first 3 cleaned tips
        # return response.text.strip().split("\n")
    except Exception as e:
        st.warning("⚠️ Gemini API Error. Showing default tips.")
        return ["Stay safe on the road! 🚗", "Watch for changing weather! 🌦️", "Keep emergency kit handy! 🧰"]

def predict_best_route(start, end, traffic, base_time):
    routes = df[(df['Start'] == start) & (df['End'] == end)]
    if routes.empty:
        st.error("❌ No routes found between these locations.")
        return

    best_route = routes.loc[routes['Time_Minutes'].idxmin()]
    checkpoints = best_route['Checkpoints'].split(', ')

    weather_impacts = []
    st.subheader("🌦️ Weather Impact")

    for location in [start] + checkpoints + [end]:
        impact, temp, wind, rain = get_weather_impact(location)
        weather_impacts.append(impact)

        if temp is not None:
            with st.expander(f"📍 {location} — {impact:.2f} impact (Temp: {temp}°C, Wind: {wind} km/h, Rain: {rain} mm)"):
                tips = get_ai_recommendations(temp, wind, rain)
                for tip in tips:
                    st.markdown(f"- {tip}")
        else:
            st.warning(f"⚠️ Could not fetch weather for {location}")

    avg_weather_impact = np.mean(weather_impacts)

    model = joblib.load("final.pkl")
    predicted_time = model.predict([[best_route['Distance_km'], traffic, base_time, avg_weather_impact]])[0]
    predicted_time = max(base_time, predicted_time)

    st.subheader("🚀 Best Route Prediction")
    st.markdown(f"**🛣️ Path:** {best_route['Checkpoints']}")
    st.markdown(f"**📏 Distance:** {best_route['Distance_km']} km")
    st.markdown(f"**⏳ Predicted Time:** `{predicted_time:.2f}` minutes")

# --- Streamlit UI ---

st.title("🚗 Trackease: Simplifying Travel by Finding You the Best Routes")
st.markdown("Get weather-aware, AI-enhanced travel estimates for routes across India.")

start = st.selectbox("📍 Start Location", sorted(df['Start'].unique()))
end = st.selectbox("🏁 Destination", sorted(df['End'].unique()))
traffic = st.slider("🚦 Traffic Level", 1.0, 1.5, step=0.05, value=1.2)
base_time = st.number_input("⏱️ Base Travel Time (minutes)", min_value=1.0, value=30.0)

if st.button("Predict Best Route"):
    predict_best_route(start, end, traffic, base_time)

# --- Model Metrics ---
st.markdown("---")
# st.subheader("📊 Model Performance")
y_pred = model.predict(X_test)
# st.write(f"**✅ MAE:** {mean_absolute_error(y_test, y_pred):.2f} minutes")
# st.write(f"**✅ MSE:** {mean_squared_error(y_test, y_pred):.2f}")
# st.write(f"**✅ RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
# st.write(f"**✅ R² Score:** {r2_score(y_test, y_pred):.4f}")
