import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

st.set_page_config(page_title="FarmIQ - Crop Yield Predictor", page_icon="🌾", layout="wide")

# Safe image (404 nahi aayega)
try:
    st.image("https://img.freepik.com/premium-vector/modern-farm-logo-vector_658271-1527.jpg?w=360", width=80)
except:
    st.title("🌾 FarmIQ")

st.title("🌾 Crop Yield Prediction")
st.markdown("---")

# ---------- Load Models ----------
base_dir = os.path.dirname(os.path.abspath(__file__))

def load_file(filename):
    paths = [
        os.path.join(base_dir, 'models', filename),
        os.path.join(base_dir, 'src', 'models', filename),
        os.path.join(base_dir, filename),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except:
                import pickle
                with open(p, 'rb') as f:
                    return pickle.load(f)
    raise FileNotFoundError(f"{filename} not found")

try:
    model = load_file('crop_yield_model.pkl')
    label_encoder = load_file('label_encoder.pkl')
    scaler = load_file('scaler.pkl')
    st.success("✅ Models Loaded")
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

# ---------- Inputs ----------
col1, col2 = st.columns(2)
with col1:
    nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
    phosphorus = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0)
    potassium = st.number_input("Potassium (K)", 0.0, 200.0, 50.0)
    temperature = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    ph = st.number_input("pH Level", 0.0, 14.0, 7.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
    crop = st.selectbox("Crop Type", label_encoder.classes_ if hasattr(label_encoder, 'classes_') else ["Wheat","Rice","Corn"])

if st.button("Predict Yield"):
    try:
        crop_enc = label_encoder.transform([crop])[0]
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, crop_enc]])
        try:
            features = scaler.transform(features)
        except:
            pass
        pred = model.predict(features)[0]
        st.success(f"🌾 Predicted Yield: {pred:.2f} tons/ha")
    except Exception as e:
        st.error(f"❌ {e}")
        st.text(traceback.format_exc())
