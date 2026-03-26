# 🌾 Smart Agri Advisor - Crop Yield Prediction with Profit Optimization

## Overview:
Smart Agri Advisor is an AI-powered web application that helps farmers predict crop yields based on soil parameters (N, P, K, pH), weather conditions (temperature, humidity, rainfall), and crop type. It provides fertilizer recommendations, profit analysis, and ROI calculations to maximize farming productivity and profitability.

## Objectives:
- Predict crop yield accurately using machine learning models
- Recommend optimal crops based on soil and environmental conditions
- Calculate expected profit with cost and revenue analysis
- Provide fertilizer suggestions for soil deficiencies
- Generate monthly and yearly trend visualizations for better planning

## Features:
- 🌾 **Yield Prediction** - Predicts yield in tons/hectare using ML models
- 💰 **Profit Calculator** - Estimates revenue, input costs, and net profit with ROI
- 🌱 **Fertilizer Recommendations** - Suggests nutrients based on soil deficiency
- 📊 **Crop Comparison** - Compares yield and profit across multiple crops
- 📈 **Visual Analytics** - Correlation heatmaps, feature importance, seasonal trends
- 🤖 **Multiple ML Models** - Linear Regression, Random Forest, XGBoost, Ensemble

## Technologies/Tools Used:
| Category | Technologies |
|----------|-------------|
| Programming | Python 3.8+ |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Model Deployment | Joblib |
| Version Control | Git, GitHub |

## Steps to Install & Run the Project:

### Prerequisites
- Python 3.8 or higher installed
- Git installed

### Installation

**1. Clone the repository**  
bash  
git clone https://github.com/your-username/smart-agri-advisor.git  
cd smart-agri-advisor  

**2. Create virtual environment**  

bash  
python -m venv venv  
venv\Scripts\activate  

**3. Install dependencies**  
pip install -r requirements.txt  

**4. Generate dataset**  
python data/raw/create_dataset.py  

**5. Train models**  

python -c "from src.models import CropYieldModel; from src.data_loader import load_data, get_train_test_split; df = load_data(); X_train, X_test, y_train, y_test, scaler, le = get_train_test_split(df); model = CropYieldModel(); model.train_all(X_train, y_train); model.save_best_model()"  

**6. Run the application**  

bash  
streamlit run app.py  

**7. Open browser and navigate**  

# **Instructions for Testing:**  

**1. Yield Prediction Test**  

Enter soil parameters (N, P, K, pH)    
Enter weather data (temperature, humidity, rainfall)  
Select crop type  
Click "Predict Yield" - verify output shows yield in tons/hectare  

**2. Profit Analysis Test**  

Input land area and cost parameters (seeds, fertilizer, labor)    
View revenue, total cost, net profit, and ROI percentage    
Verify calculations are accurate  

**3. Model Accuracy Test**  

Check model performance metrics in analytics section    
Expected R² score: >0.90 for trained models    
Compare predictions with known values    

**4. Visualization Test**  

Navigate to Analytics tab    
Verify correlation heatmap displays  
Check feature importance graph  
Ensure all plots render correctly  

**5.Error Handling Test**

Submit empty fields - should show validation messages   
Enter invalid values - should handle gracefully  

📧 For queries, contact: [anshu.25bai11353@vitbhopal.ac.in]  

⭐ Star this repository if you find it helpful!  

# Image of Working Model

<img width="1470" height="956" alt="Screenshot 2026-03-26 at 4 00 52 PM" src="https://github.com/user-attachments/assets/cbd9cd2c-ddb3-4778-92e4-165b9e24ea7b" />

<img width="1470" height="956" alt="Screenshot 2026-03-26 at 4 01 02 PM" src="https://github.com/user-attachments/assets/7407e250-c859-41d2-91b0-0d502d497fa7" />

<img width="1470" height="956" alt="Screenshot 2026-03-26 at 4 01 30 PM" src="https://github.com/user-attachments/assets/1da461b0-d699-4323-a3f3-b1a373281db3" />

<img width="1470" height="956" alt="Screenshot 2026-03-26 at 4 02 07 PM" src="https://github.com/user-attachments/assets/ab95bda0-ed1b-47a9-9973-f33c59b38cec" />

<img width="1470" height="956" alt="Screenshot 2026-03-26 at 4 02 19 PM" src="https://github.com/user-attachments/assets/46595e34-ca90-4cb6-b1e1-89f9f5bcff51" />

<img width="1470" height="956" alt="Screenshot 2026-03-26 at 4 02 32 PM" src="https://github.com/user-attachments/assets/d821200a-431c-45d0-9a0c-45f5e766fac7" />

<img width="1470" height="956" alt="Screenshot 2026-03-26 at 4 04 09 PM" src="https://github.com/user-attachments/assets/f891e246-7b08-4b1a-993b-57883bcea55f" />

