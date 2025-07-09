# car_price_predictor.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

# Constants
DATA_PATH = "car data.csv"  
CURRENT_YEAR = datetime.now().year
MODEL_PATH = "car_price_model.pkl"

def load_data():
    """Load the car dataset"""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Dataset loaded from {DATA_PATH} ({len(df)} records)")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {DATA_PATH}")
        print("Please make sure the CSV file exists in the same folder as this script.")
        return None

def preprocess_data(df):
    """Preprocess data and create features"""
    # Feature engineering
    df['Car_Age'] = CURRENT_YEAR - df['Year']
    
    # Drop unnecessary columns
    df = df.drop(['Car_Name'], axis=1)
    
    # Define features and target
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    
    return X, y

def train_model():
    """Train and save the price prediction model"""
    df = load_data()
    if df is None:
        return None
    
    X, y = preprocess_data(df)
    
    # Define preprocessing for categorical columns
    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    numerical_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner', 'Car_Age']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])
    
    # Create pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("⚙️ Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"📊 Model Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved as '{MODEL_PATH}'")
    
    return model

def predict_price(input_data):
    """Predict car price from input features"""
    # Load or train model
    try:
        model = joblib.load(MODEL_PATH)
        print("💡 Using existing model")
    except FileNotFoundError:
        print("⚠️ Model not found. Training new model...")
        model = train_model()
        if model is None:
            return None
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # ADD MISSING FEATURE: Car Age
    input_df['Car_Age'] = CURRENT_YEAR - input_df['Year']
    
    # Make prediction
    predicted_price = model.predict(input_df)[0]
    
    # Show results
    print("\n🔮 Prediction Results:")
    print(f"Predicted Selling Price: ₹{predicted_price:.2f} Lakhs")
    
    return predicted_price

def main():
    """Main function for user interaction"""
    print("="*50)
    print("Car Price Prediction System")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("1. Train new model")
        print("2. Predict car price")
        print("3. Exit")
        
        choice = input("Choose an option (1-3): ")
        
        if choice == '1':
            print("\nTraining new model...")
            train_model()
            
        elif choice == '2':
            print("\nEnter car details:")
            try:
                year = int(input("Manufacturing Year: "))
                present_price = float(input("Current Showroom Price (Lakhs): "))
                driven_kms = float(input("Driven Kilometers: "))
                fuel_type = input("Fuel Type (Petrol/Diesel/CNG): ").capitalize()
                selling_type = input("Seller Type (Dealer/Individual): ").capitalize()
                transmission = input("Transmission (Manual/Automatic): ").capitalize()
                owner = int(input("Number of Previous Owners (0-3): "))
                
                input_data = {
                    'Year': year,
                    'Present_Price': present_price,
                    'Driven_kms': driven_kms,
                    'Fuel_Type': fuel_type,
                    'Selling_type': selling_type,
                    'Transmission': transmission,
                    'Owner': owner
                }
                
                predict_price(input_data)
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
                print("Please enter correct values.")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                
        elif choice == '3':
            print("Exiting program...")
            break
            
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()