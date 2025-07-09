# iris_predictor.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "iris.csv"  

def load_data():
    """Load the Iris dataset"""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Dataset loaded from {DATA_PATH}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {DATA_PATH}")
        print("Please make sure the CSV file exists in the same folder as this script.")
        exit(1)

def train_model():
    """Train and save the classification model"""
    df = load_data()
    
    # Prepare features and target
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, 'iris_model.pkl')
    print("💾 Model trained and saved as 'iris_model.pkl'")
    
    return model

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """Predict iris species from measurements"""
    # Load or train model
    try:
        model = joblib.load('iris_model.pkl')
        print("💡 Using existing model")
    except FileNotFoundError:
        print("⚠️ Model not found. Training new model...")
        model = train_model()
    
    # Make prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    
    # Get probabilities
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    
    # Show results
    print("\n🔮 Prediction Results:")
    print(f"Species: {prediction}")
    print("Probabilities:")
    for cls, prob in zip(classes, probabilities):
        print(f"  {cls}: {prob:.1%}")
    
    return prediction

def main():
    """Main function for user interaction"""
    print("="*50)
    print("Iris Flower Species Predictor")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("1. Train new model")
        print("2. Predict species")
        print("3. Exit")
        
        choice = input("Choose an option (1-3): ")
        
        if choice == '1':
            print("\nTraining new model...")
            train_model()
            
        elif choice == '2':
            print("\nEnter flower measurements in centimeters:")
            try:
                sepal_length = float(input("Sepal Length: "))
                sepal_width = float(input("Sepal Width: "))
                petal_length = float(input("Petal Length: "))
                petal_width = float(input("Petal Width: "))
                
                predict_species(sepal_length, sepal_width, petal_length, petal_width)
            except ValueError:
                print("❌ Invalid input. Please enter numbers only.")
                
        elif choice == '3':
            print("Exiting program...")
            break
            
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()


    