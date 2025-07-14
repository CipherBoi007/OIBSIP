import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Advertising.csv').iloc[:, 1:]  


X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = SVR(kernel='rbf', C=1.0, epsilon=0.1)    
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

    # Print performance metrics
print("\nPerformance Metrics (SVR Model):")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Save model and scaler for Streamlit app
import joblib
joblib.dump(model, 'sales_model.pkl')
joblib.dump(scaler, 'sales_scaler.pkl')
print("\nModel and scaler saved as 'sales_model.pkl' and 'sales_scaler.pkl'.")



    # The Streamlit app will handle user input and prediction.