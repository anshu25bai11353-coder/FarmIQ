print("hi")

# Datasets 

import pandas as pd
import numpy as np
import random
import os

os.makedirs('data/raw', exist_ok=True)

np.random.seed(42)
random.seed(42)

crops = {
    'Rice': {'N': 80, 'P': 40, 'K': 40, 'pH': 6.5, 'temp': 28, 'humidity': 80, 'rainfall': 200, 'yield_factor': 3.5},
    'Wheat': {'N': 120, 'P': 60, 'K': 60, 'pH': 7.0, 'temp': 20, 'humidity': 60, 'rainfall': 150, 'yield_factor': 3.0},
    'Maize': {'N': 150, 'P': 70, 'K': 50, 'pH': 6.8, 'temp': 25, 'humidity': 65, 'rainfall': 180, 'yield_factor': 4.0},
    'Sugarcane': {'N': 200, 'P': 100, 'K': 100, 'pH': 7.2, 'temp': 30, 'humidity': 75, 'rainfall': 250, 'yield_factor': 8.0},
    'Cotton': {'N': 100, 'P': 50, 'K': 70, 'pH': 7.0, 'temp': 28, 'humidity': 55, 'rainfall': 120, 'yield_factor': 2.5},
    'Groundnut': {'N': 30, 'P': 50, 'K': 40, 'pH': 6.2, 'temp': 27, 'humidity': 60, 'rainfall': 100, 'yield_factor': 1.8},
    'Soybean': {'N': 80, 'P': 60, 'K': 50, 'pH': 6.5, 'temp': 26, 'humidity': 70, 'rainfall': 130, 'yield_factor': 2.2},
    'Potato': {'N': 150, 'P': 80, 'K': 100, 'pH': 6.0, 'temp': 18, 'humidity': 75, 'rainfall': 150, 'yield_factor': 2.5},
    'Onion': {'N': 100, 'P': 50, 'K': 50, 'pH': 6.8, 'temp': 22, 'humidity': 65, 'rainfall': 100, 'yield_factor': 2.0},
    'Tomato': {'N': 120, 'P': 60, 'K': 80, 'pH': 6.5, 'temp': 24, 'humidity': 70, 'rainfall': 120, 'yield_factor': 3.0}
}

n_samples = 5000
data = []

for _ in range(n_samples):
    crop = random.choice(list(crops.keys()))
    optimal = crops[crop]
    
    N = max(0, optimal['N'] + np.random.normal(0, 30))
    P = max(0, optimal['P'] + np.random.normal(0, 20))
    K = max(0, optimal['K'] + np.random.normal(0, 20))
    pH = optimal['pH'] + np.random.normal(0, 0.5)
    temperature = optimal['temp'] + np.random.normal(0, 5)
    humidity = optimal['humidity'] + np.random.normal(0, 10)
    rainfall = optimal['rainfall'] + np.random.normal(0, 50)
    
    distance = (
        abs(N - optimal['N']) / optimal['N'] * 0.3 +
        abs(P - optimal['P']) / optimal['P'] * 0.2 +
        abs(K - optimal['K']) / optimal['K'] * 0.2 +
        abs(pH - optimal['pH']) / optimal['pH'] * 0.1 +
        abs(temperature - optimal['temp']) / optimal['temp'] * 0.1 +
        abs(humidity - optimal['humidity']) / optimal['humidity'] * 0.05 +
        abs(rainfall - optimal['rainfall']) / optimal['rainfall'] * 0.05
    )
    
    base_yield = optimal['yield_factor'] * (1 - min(0.7, distance * 0.7))
    yield_tons = base_yield + np.random.normal(0, 0.2)
    yield_tons = max(0.5, min(yield_tons, optimal['yield_factor'] * 1.2))
    
    data.append({
        'Crop': crop,
        'N': round(N, 2),
        'P': round(P, 2),
        'K': round(K, 2),
        'pH': round(pH, 2),
        'Temperature': round(temperature, 2),
        'Humidity': round(humidity, 2),
        'Rainfall': round(rainfall, 2),
        'Yield_tons_per_hectare': round(yield_tons, 2)
    })

df = pd.DataFrame(data)
df.to_csv('data/raw/crop_yield_dataset.csv', index=False)
print(f"Dataset created with {len(df)} samples")
print(df.head())


# src

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(filepath='data/raw/crop_yield_dataset.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):
    df_processed = df.copy()
    le = LabelEncoder()
    df_processed['Crop_Encoded'] = le.fit_transform(df_processed['Crop'])
    
    feature_cols = ['N', 'P', 'K', 'pH', 'Temperature', 'Humidity', 'Rainfall', 'Crop_Encoded']
    X = df_processed[feature_cols]
    y = df_processed['Yield_tons_per_hectare']
    
    return X, y, le

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def get_train_test_split(df, test_size=0.2, random_state=42):
    X, y, le = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le


# src2

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class CropYieldModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        
    def train_all(self, X_train, y_train):
        print("=" * 50)
        print("Training Models...")
        print("=" * 50)
        
        self.models['Random Forest'] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        self.models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]
        self.models['Ensemble'] = VotingRegressor(estimators)
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            
            print(f"✅ {name:20s} | R²: {train_r2:.4f} | RMSE: {train_rmse:.4f}")
            
            if train_r2 > self.best_score:
                self.best_score = train_r2
                self.best_model = model
                self.best_model_name = name
        
        print("=" * 50)
        print(f"🏆 Best Model: {self.best_model_name} with R²: {self.best_score:.4f}")
        return self.models
    
    def evaluate(self, X_test, y_test):
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results.append({
                'Model': name,
                'R² Score': round(r2, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4)
            })
        return pd.DataFrame(results).sort_values('R² Score', ascending=False)
    
    def save_best_model(self, filepath='models/crop_yield_model.pkl'):
        joblib.dump(self.best_model, filepath)
        print(f"Best model saved to {filepath}")
    
    def load_model(self, filepath='models/crop_yield_model.pkl'):
        self.best_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.best_model
    

    #src/profit_calculator.py

class ProfitCalculator:
    
    crop_prices = {
        'Rice': 22000, 'Wheat': 24000, 'Maize': 20000, 'Sugarcane': 3500,
        'Cotton': 60000, 'Groundnut': 55000, 'Soybean': 45000,
        'Potato': 15000, 'Onion': 18000, 'Tomato': 12000
    }
    
    input_costs = {
        'Rice': {'seeds': 4000, 'fertilizer': 5000, 'pesticides': 3000, 'labor': 8000, 'irrigation': 2000},
        'Wheat': {'seeds': 3500, 'fertilizer': 4500, 'pesticides': 2500, 'labor': 7000, 'irrigation': 1500},
        'Maize': {'seeds': 3000, 'fertilizer': 4000, 'pesticides': 2000, 'labor': 6000, 'irrigation': 1500},
        'Sugarcane': {'seeds': 8000, 'fertilizer': 10000, 'pesticides': 5000, 'labor': 15000, 'irrigation': 3000},
        'Cotton': {'seeds': 5000, 'fertilizer': 6000, 'pesticides': 4000, 'labor': 10000, 'irrigation': 2000},
        'Groundnut': {'seeds': 4000, 'fertilizer': 3500, 'pesticides': 2000, 'labor': 5000, 'irrigation': 1000},
        'Soybean': {'seeds': 3500, 'fertilizer': 4000, 'pesticides': 2500, 'labor': 6000, 'irrigation': 1200},
        'Potato': {'seeds': 6000, 'fertilizer': 5000, 'pesticides': 3000, 'labor': 8000, 'irrigation': 1500},
        'Onion': {'seeds': 5000, 'fertilizer': 4500, 'pesticides': 3000, 'labor': 7000, 'irrigation': 1200},
        'Tomato': {'seeds': 4000, 'fertilizer': 5000, 'pesticides': 3500, 'labor': 8000, 'irrigation': 1500}
    }
    
    def __init__(self, land_area=1):
        self.land_area = land_area
    
    def calculate_total_cost(self, crop):
        costs = self.input_costs.get(crop, {'seeds': 4000, 'fertilizer': 4000, 'pesticides': 2000, 'labor': 5000, 'irrigation': 1000})
        total = sum(costs.values()) * self.land_area
        return total
    
    def calculate_profit(self, crop, yield_tons):
        price = self.crop_prices.get(crop, 20000)
        total_yield = yield_tons * self.land_area
        revenue = total_yield * price
        cost = self.calculate_total_cost(crop)
        profit = revenue - cost
        
        return {
            'crop': crop,
            'yield_tons_per_hectare': yield_tons,
            'total_yield_tons': total_yield,
            'market_price': price,
            'revenue': revenue,
            'total_cost': cost,
            'profit': profit,
            'roi_percentage': (profit / cost) * 100 if cost > 0 else 0
        }
    
    def get_fertilizer_recommendation(self, N, P, K, pH):
        recs = []
        if N < 80:
            recs.append("🌱 Add Nitrogen: Apply Urea or DAP")
        elif N > 200:
            recs.append("⚠️ Nitrogen high: Reduce application")
        else:
            recs.append("✅ Nitrogen level optimal")
        
        if P < 40:
            recs.append("🌱 Add Phosphorus: Apply SSP or DAP")
        elif P > 100:
            recs.append("⚠️ Phosphorus high: Reduce application")
        else:
            recs.append("✅ Phosphorus level optimal")
        
        if K < 40:
            recs.append("🌱 Add Potassium: Apply MOP")
        elif K > 150:
            recs.append("⚠️ Potassium high: Reduce application")
        else:
            recs.append("✅ Potassium level optimal")
        
        if pH < 5.5:
            recs.append("⚖️ Soil too acidic: Apply lime")
        elif pH > 7.5:
            recs.append("⚖️ Soil too alkaline: Apply gypsum")
        else:
            recs.append("✅ Soil pH optimal")
        
        return recs