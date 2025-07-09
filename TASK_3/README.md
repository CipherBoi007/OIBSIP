Overview

This Python application predicts the selling price of used cars based on various features like manufacturing year, current showroom price, mileage, fuel type, and more. Using machine learning with Gradient Boosting Regression, it provides accurate price estimates to help both buyers and sellers in the used car market.

Features

🚗 Accurate price predictions based on car specifications

📊 Interactive command-line interface for easy use

🤖 Automatic model training when needed

💾 Model persistence - trains once, predicts many times

📈 Performance metrics - shows R² score and RMSE after training

🔄 Real-time feature calculation - automatically calculates car age

Usage

Run the application:

/bash
python car_price_predictor.py

Example

==================================================
Car Price Prediction System
==================================================

Options:
1. Train new model
2. Predict car price
3. Exit
Choose an option (1-3): 1

Training new model...
✅ Dataset loaded from car data.csv (301 records)
⚙️ Training model...
📊 Model Performance:
  R² Score: 0.9700
  RMSE: 0.8319
💾 Model saved as 'car_price_model.pkl'

Options:
1. Train new model
2. Predict car price
3. Exit
Choose an option (1-3): 2 

Enter car details:
Manufacturing Year: 2020
Current Showroom Price (Lakhs): 16
Driven Kilometers: 2000
Fuel Type (Petrol/Diesel/CNG): petrol
Seller Type (Dealer/Individual): Dealer
Transmission (Manual/Automatic): manual
Number of Previous Owners (0-3): 2
💡 Using existing model

🔮 Prediction Results:
Predicted Selling Price: ₹13.45 Lakhs


Model Details

Algorithm: Gradient Boosting Regressor

Key Features:

Present Price (current showroom price)
Car Age (calculated as current year - manufacturing year)
Driven Kilometers
Fuel Type (one-hot encoded)
Seller Type (one-hot encoded)
Transmission Type (one-hot encoded)
Number of Previous Owners
Performance Metrics:
R² Score: 0.92-0.94
RMSE: 0.82-0.87