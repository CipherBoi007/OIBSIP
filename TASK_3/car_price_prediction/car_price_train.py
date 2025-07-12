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

DATA_PATH = "car data.csv"
CURRENT_YEAR = datetime.now().year
MODEL_PATH = "car_price_model.pkl"

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Dataset loaded from {DATA_PATH} ({len(df)} records)")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {DATA_PATH}")
        print("Please make sure the CSV file exists in the same folder as this script.")
        return None

def preprocess_data(df):
    df['Car_Age'] = CURRENT_YEAR - df['Year']
    df = df.drop(['Car_Name'], axis=1)
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    return X, y

def train_model():
    df = load_data()
    if df is None:
        return None
    X, y = preprocess_data(df)
    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    numerical_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner', 'Car_Age']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("⚙️ Training model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"📊 Model Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved as '{MODEL_PATH}'")
    return model

if __name__ == "__main__":
    train_model()
