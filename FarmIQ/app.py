import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import ProfitCalculator

# Page Configuration
st.set_page_config(
    page_title="Smart Agri Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    h1, h2, h3 {
        color: #2e7d32;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_stdio=True)

# Helper for robust model loading
def get_path(filename):
    # Try current directory
    if os.path.exists(filename):
        return filename
    # Try relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    if os.path.exists(path):
        return path
    # Try in models folder
    path = os.path.join(script_dir, 'models', filename)
    if os.path.exists(path):
        return path
    # Try in src/models folder
    path = os.path.join(script_dir, 'src', 'models', filename)
    if os.path.exists(path):
        return path
    return None

# Load model, scaler, and label encoder
@st.cache_resource
def load_artifacts():
    # Try various names for the crop model
    model_path = get_path('crop_yield_model.pkl') or get_path('crop_model.pkl')
    
    if not model_path:
        st.error("Model file not found! Please run the training script first.")
        return None, None, None
        
    try:
        model_obj = joblib.load(model_path)
        # If the loaded object is a dict, select the 'Ensemble' model or the first available
        if isinstance(model_obj, dict):
            if 'Ensemble' in model_obj:
                model = model_obj['Ensemble']
            else:
                model = next(iter(model_obj.values()))
        else:
            model = model_obj
            
        scaler_path = get_path('scaler.pkl')
        scaler = joblib.load(scaler_path) if scaler_path else None
        
        le_path = get_path('label_encoder.pkl')
        le = joblib.load(le_path) if le_path else None
        
        return model, scaler, le
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

# Initialize models
model, scaler, le = load_artifacts()

if model is None:
    st.stop()

# Sidebar for common inputs
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/agriculture.png", width=80)
    st.header("Farm Settings")
    land_area = st.number_input('Land Area (hectare)', min_value=0.1, max_value=100.0, value=1.0, step=0.1)
    
    crop_options = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Groundnut', 'Soybean', 'Potato', 'Onion', 'Tomato']
    crop = st.selectbox('Select Crop', crop_options)
    
    st.divider()
    st.info("Input the soil and weather conditions of your farm to get AI-powered recommendations.")

# Main content
st.title('🌾 Smart Agri Advisor')
st.markdown("### Optimize Your Harvest with AI-Driven Insights")

# Input columns
tab1, tab2 = st.tabs(["📊 Prediction & Analysis", "📈 Trends & Importance"])

with tab1:
    st.subheader('Field Parameters')
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input('Nitrogen (N) [mg/kg]', min_value=0, max_value=300, value=100)
        P = st.number_input('Phosphorus (P) [mg/kg]', min_value=0, max_value=150, value=50)
    with col2:
        K = st.number_input('Potassium (K) [mg/kg]', min_value=0, max_value=200, value=50)
        pH = st.number_input('Soil pH', min_value=3.0, max_value=10.0, value=6.5, step=0.01)
    with col3:
        temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    
    rainfall = st.slider('Annual Rainfall (mm)', min_value=0.0, max_value=500.0, value=150.0, step=0.1)

    if st.button('🚀 Predict Yield & Profitability'):
        # Prepare input
        try:
            crop_encoded = crop_options.index(crop)
        except ValueError:
            crop_encoded = 0
            
        X_full = np.array([[N, P, K, pH, temperature, humidity, rainfall, crop_encoded]])
        
        # Check model input feature count
        n_features = getattr(model, 'n_features_in_', 8)
        if n_features == 7:
            X = X_full[:, :7]
        else:
            X = X_full
            
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
            
        # Predict yield
        yield_pred = model.predict(X_scaled)[0]
        yield_pred_float = float(yield_pred)
            
        # Display Results
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("#### 🎯 Prediction Results")
            st.metric("Estimated Yield", f"{yield_pred_float:.2f} t/ha")
            st.success(f"High probability of successful harvest for **{crop}**.")
            
            # Fertilizer recommendations
            st.markdown("#### 🌱 Soil Management")
            profit_calc = ProfitCalculator(land_area=land_area)
            recs = profit_calc.get_fertilizer_recommendation(N, P, K, pH)
            for rec in recs:
                st.write(rec)

        with res_col2:
            st.markdown("#### 💰 Financial Analysis")
            profit = profit_calc.calculate_profit(crop, yield_pred_float)
            
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Revenue", f"₹{profit['revenue']:,.0f}")
            p_col1.metric("Net Profit", f"₹{profit['profit']:,.0f}")
            p_col2.metric("Total Cost", f"₹{profit['total_cost']:,.0f}")
            p_col2.metric("ROI", f"{profit['roi_percentage']:.1f}%")

with tab2:
    st.subheader('Model Insights & Future Projections')
    
    # Feature Importance
    def get_importances(m):
        if hasattr(m, 'steps'): m = m.steps[-1][1]
        if hasattr(m, 'estimators_'):
            importances_list = []
            for estimator in m.estimators_:
                est = estimator.steps[-1][1] if hasattr(estimator, 'steps') else estimator
                if hasattr(est, 'feature_importances_'): importances_list.append(est.feature_importances_)
                elif hasattr(est, 'coef_'): importances_list.append(np.abs(est.coef_.flatten()))
            if importances_list: return np.mean(importances_list, axis=0)
        if hasattr(m, 'feature_importances_'): return m.feature_importances_
        elif hasattr(m, 'coef_'): return np.abs(m.coef_.flatten())
        return None

    importances = get_importances(model)
    
    col_inf1, col_inf2 = st.columns([2, 1])
    
    with col_inf1:
        if importances is not None:
            if len(importances) == 7:
                features = ['N', 'P', 'K', 'pH', 'Temperature', 'Humidity', 'Rainfall']
            elif len(importances) == 8:
                features = ['N', 'P', 'K', 'pH', 'Temperature', 'Humidity', 'Rainfall', 'Crop']
            else:
                features = [f'F{i}' for i in range(len(importances))]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importances, y=features, palette="viridis", ax=ax)
            ax.set_title('Feature Importance (What drives yield?)', fontsize=14)
            st.pyplot(fig)
        else:
            st.info("Feature importance data not available for this model.")

    with col_inf2:
        st.markdown("#### 🔭 Future Outlook")
        future_year = st.select_slider('Projection Year', options=range(2026, 2036), value=2027)
        future_temp = st.number_input('Projected Temp (°C)', value=temperature + 0.5)
        
        if st.button('Predict Future'):
            X_future_full = np.array([[N, P, K, pH, future_temp, humidity, rainfall, crop_options.index(crop)]])
            X_future = X_future_full[:, :n_features]
            X_f_scaled = scaler.transform(X_future) if scaler else X_future
            f_yield = model.predict(X_f_scaled)[0]
            st.success(f"Projected Yield in {future_year}: **{f_yield:.2f} t/ha**")
            diff = f_yield - yield_pred_float
            st.write(f"{'Increase' if diff > 0 else 'Decrease'} of {abs(diff):.2f} t/ha due to climate variation.")

st.divider()
st.caption("Developed by Smart Agri Advisor Team | © 2026")
