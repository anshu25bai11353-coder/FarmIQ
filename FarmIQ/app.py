import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import ProfitCalculator, load_data

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
    """, unsafe_allow_html=True)

# Helper for robust model loading
def get_path(filename):
    if os.path.exists(filename):
        return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    if os.path.exists(path):
        return path
    path = os.path.join(script_dir, 'models', filename)
    if os.path.exists(path):
        return path
    path = os.path.join(script_dir, 'src', 'models', filename)
    if os.path.exists(path):
        return path
    return None

# Load artifacts
@st.cache_resource
def load_artifacts():
    model_path = get_path('crop_yield_model.pkl') or get_path('crop_model.pkl')
    if not model_path:
        st.error("Model file not found! Please run the training script first.")
        return None, None, None
    try:
        model_obj = joblib.load(model_path)
        if isinstance(model_obj, dict):
            model = model_obj['Ensemble'] if 'Ensemble' in model_obj else next(iter(model_obj.values()))
        else:
            model = model_obj
        scaler = joblib.load(get_path('scaler.pkl')) if get_path('scaler.pkl') else None
        le = joblib.load(get_path('label_encoder.pkl')) if get_path('label_encoder.pkl') else None
        return model, scaler, le
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

# Initialize session state
if 'yield_pred' not in st.session_state: st.session_state.yield_pred = 0.0

model, scaler, le = load_artifacts()
if model is None: st.stop()

# Load Dataset for Heatmap
@st.cache_data
def get_dataset():
    try:
        return load_data()
    except:
        return None

df_raw = get_dataset()

# Sidebar
with st.sidebar:
    st.image("https://img.freepik.com/premium-vector/modern-farm-logo-vector_658271-1527.jpg?w=360", width=80)
    st.header("Farm Settings")
    land_area = st.number_input('Land Area (hectares)', min_value=0.1, max_value=100.0, value=1.0, step=0.1)
    crop_options = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Groundnut', 'Soybean', 'Potato', 'Onion', 'Tomato']
    crop = st.selectbox('Select Crop', crop_options)
    st.divider()
    st.info("AI-Powered Agriculture Insight System")

# Main
st.title('🌾 Smart Agri Advisor')
tab1, tab2 = st.tabs(["📊 Prediction & Analysis", "📈 Trends & Importance"])

with tab1:
    st.subheader('Field Parameters')
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input('Nitrogen (N)', min_value=0.0, max_value=300.0, value=100.0)
        P = st.number_input('Phosphorus (P)', min_value=0.0, max_value=150.0, value=50.0)
    with col2:
        K = st.number_input('Potassium (K)', min_value=0.0, max_value=200.0, value=50.0)
        pH = st.number_input('Soil pH', min_value=3.0, max_value=10.0, value=6.5, step=0.1)
    with col3:
        temperature = st.number_input('Temp (°C)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    rainfall = st.slider('Annual Rainfall (mm)', 0.0, 500.0, 150.0)

    # Core Prediction Logic
    try:
        crop_encoded = crop_options.index(crop)
    except:
        crop_encoded = 0
    
    n_features = getattr(model, 'n_features_in_', 8)
    X_raw = np.array([[N, P, K, pH, temperature, humidity, rainfall, crop_encoded]])
    X_input = X_raw[:, :n_features]
    X_scaled = scaler.transform(X_input) if scaler else X_input
    
    st.session_state.yield_pred = float(model.predict(X_scaled)[0])

    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown("#### 🎯 Yield Forecast")
        st.metric("Estimated Yield", f"{st.session_state.yield_pred:.2f} t/ha")
        st.success(f"Optimized conditions identified for **{crop}**.")
        profit_calc = ProfitCalculator(land_area=land_area)
        for rec in profit_calc.get_fertilizer_recommendation(N, P, K, pH):
            st.write(rec)

    with res_col2:
        st.markdown("#### 💰 Financial Analysis")
        p = profit_calc.calculate_profit(crop, st.session_state.yield_pred)
        c1, c2 = st.columns(2)
        c1.metric("Revenue", f"₹{p['revenue']:,.0f}")
        c1.metric("Net Profit", f"₹{p['profit']:,.0f}")
        c2.metric("Total Cost", f"₹{p['total_cost']:,.0f}")
        c2.metric("ROI", f"{p['roi_percentage']:.1f}%")

with tab2:
    st.subheader('Interactive Model Insights')
    
    # Sensitivity Plot with Plotly
    def plot_sensitivity_interactive(base_X, feat_idx, feat_name, r_vals):
        y_vals = []
        for v in r_vals:
            X_t = base_X.copy()
            X_t[0, feat_idx] = v
            X_ts = scaler.transform(X_t[:, :n_features]) if scaler else X_t[:, :n_features]
            y_vals.append(model.predict(X_ts)[0])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r_vals, y=y_vals, mode='lines+markers', name='Impact', line=dict(color='#2e7d32', width=3)))
        fig.add_vline(x=base_X[0, feat_idx], line_dash="dash", line_color="red", annotation_text="Current")
        fig.update_layout(title=f'Yield Sensitivity: {feat_name}', xaxis_title=feat_name, yaxis_title='Yield (t/ha)', hovermode='x unified', template='plotly_white')
        return fig

    c_inf1, c_inf2 = st.columns([2, 1])
    
    with c_inf1:
        st.markdown("#### 📡 Predictive Dynamics")
        s_feat = st.selectbox('Analyze impact of:', ['Rainfall', 'Temperature', 'Nitrogen', 'pH'])
        f_map = {'Rainfall': (6, np.linspace(0, 500, 50)), 'Temperature': (4, np.linspace(0, 50, 50)), 'Nitrogen': (0, np.linspace(0, 300, 50)), 'pH': (3, np.linspace(3, 10, 50))}
        idx, rng = f_map[s_feat]
        st.plotly_chart(plot_sensitivity_interactive(X_raw, idx, s_feat, rng), width='stretch')
        
        st.divider()
        st.markdown("#### 📊 Comprehensive Correlations")
        if df_raw is not None:
            # Select only numeric for correlation
            corr = df_raw.select_dtypes(include=[np.number]).corr()
            fig_h = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='Viridis', title="Feature Correlation Matrix")
            st.plotly_chart(fig_h, width='stretch')

    with c_inf2:
        st.markdown("#### ⚖️ Explaining the Model")
        # Feature Importance
        def get_fi(m):
            if hasattr(m, 'steps'): m = m.steps[-1][1]
            if hasattr(m, 'estimators_'):
                list_fi = []
                for e in m.estimators_:
                    est = e.steps[-1][1] if hasattr(e, 'steps') else e
                    if hasattr(est, 'feature_importances_'): list_fi.append(est.feature_importances_)
                if list_fi: return np.mean(list_fi, axis=0)
            return getattr(m, 'feature_importances_', None)

        fi = get_fi(model)
        if fi is not None:
            lbls = ['N', 'P', 'K', 'pH', 'Temp', 'Hum', 'Rain', 'Crop'][:len(fi)]
            df_fi = pd.DataFrame({'Feature': lbls, 'Importance': fi}).sort_values('Importance')
            fig_fi = px.bar(df_fi, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Greens', title="Global Drivers")
            st.plotly_chart(fig_fi, width='stretch')

        st.divider()
        st.markdown("#### 🔭 Future Outlook")
        f_yr = st.select_slider('Projection Year', options=range(2026, 2036), value=2027)
        f_t = st.number_input('Future Temp (°C)', value=temperature + 0.5)
        X_f = np.array([[N, P, K, pH, f_t, humidity, rainfall, crop_encoded]])
        X_fs = scaler.transform(X_f[:, :n_features]) if scaler else X_f[:, :n_features]
        fy = model.predict(X_fs)[0]
        st.metric(f"Projected Yield ({f_yr})", f"{fy:.2f} t/ha")
        delta = fy - st.session_state.yield_pred
        st.write(f"Delta: **{delta:+.2f} t/ha** vs current.")

st.divider()
st.caption("FarmIQ Smart Agri Advisor by Anshu Sharma| Optimized for High-Yield Decision Making")


