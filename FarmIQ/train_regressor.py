import os
import joblib
from src.data_loader import load_data, get_train_test_split, CropYieldModel

def main():
    os.makedirs('src/models', exist_ok=True)
    print("Loading data...")
    df = load_data()
    print("Preparing data...")
    X_train, X_test, y_train, y_test, scaler, le = get_train_test_split(df)
    print("Training regression models...")
    model = CropYieldModel()
    model.train_all(X_train, y_train)
    print("\nEvaluation Results:")
    results = model.evaluate(X_test, y_test)
    print(results.to_string(index=False))
    # Save only the best model
    joblib.dump(model.best_model, 'src/models/crop_model.pkl')
    joblib.dump(scaler, 'src/models/scaler.pkl')
    joblib.dump(le, 'src/models/label_encoder.pkl')
    print("\n✅ Regression model training complete! Files saved in 'src/models/' folder.")

if __name__ == "__main__":
    main()
