import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from datetime import datetime

MODEL_PATH = "car_price_model.pkl"
CURRENT_YEAR = datetime.now().year

# Prediction function
def predict_price(input_data):
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        messagebox.showerror("Model Error", "Model file not found. Please train the model first.")
        return None
    input_df = pd.DataFrame([input_data])
    input_df['Car_Age'] = CURRENT_YEAR - input_df['Year']
    predicted_price = model.predict(input_df)[0]
    return predicted_price

def on_predict():
    try:
        year = int(year_entry.get())
        present_price = float(price_entry.get())
        driven_kms = float(kms_entry.get())
        fuel_type = fuel_var.get()
        selling_type = selling_var.get()
        transmission = trans_var.get()
        owner = int(owner_entry.get())
        input_data = {
            'Year': year,
            'Present_Price': present_price,
            'Driven_kms': driven_kms,
            'Fuel_Type': fuel_type,
            'Selling_type': selling_type,
            'Transmission': transmission,
            'Owner': owner
        }
        price = predict_price(input_data)
        if price is not None:
            result_label.config(text=f"Predicted Selling Price: ₹{price:.2f} Lakhs", fg="#4f8ef7")
    except ValueError:
        messagebox.showwarning("Input Error", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Car Price Prediction GUI")
root.geometry("500x420")

# Light green header
root.configure(bg="#eaffea")
header = tk.Label(root, text="🚗 Car Price Predictor", font=("Segoe UI", 20, "bold"), bg="#006400", fg="white", pady=12)
header.pack(fill="x")

main_frame = tk.Frame(root, bg="#eaffea")
main_frame.pack(expand=True, fill="both", padx=20, pady=10)

# Year
year_label = tk.Label(main_frame, text="Manufacturing Year:", font=("Segoe UI", 12), bg="#f0f4fc")
year_label.grid(row=0, column=0, sticky="w", pady=5)
year_entry = tk.Entry(main_frame, font=("Segoe UI", 12), width=20)
year_entry.grid(row=0, column=1, pady=5)

# Present Price
price_label = tk.Label(main_frame, text="Current Showroom Price (Lakhs):", font=("Segoe UI", 12), bg="#f0f4fc")
price_label.grid(row=1, column=0, sticky="w", pady=5)
price_entry = tk.Entry(main_frame, font=("Segoe UI", 12), width=20)
price_entry.grid(row=1, column=1, pady=5)

# Driven Kms
kms_label = tk.Label(main_frame, text="Driven Kilometers:", font=("Segoe UI", 12), bg="#f0f4fc")
kms_label.grid(row=2, column=0, sticky="w", pady=5)
kms_entry = tk.Entry(main_frame, font=("Segoe UI", 12), width=20)
kms_entry.grid(row=2, column=1, pady=5)

# Fuel Type
fuel_label = tk.Label(main_frame, text="Fuel Type:", font=("Segoe UI", 12), bg="#f0f4fc")
fuel_label.grid(row=3, column=0, sticky="w", pady=5)
fuel_var = tk.StringVar(value="Petrol")
fuel_menu = tk.OptionMenu(main_frame, fuel_var, "Petrol", "Diesel", "CNG")
fuel_menu.config(font=("Segoe UI", 12), width=18, bg="#228B22", fg="white", activebackground="#006400", activeforeground="white", relief="groove", highlightbackground="#006400")
fuel_menu.grid(row=3, column=1, pady=5)

# Selling Type
selling_label = tk.Label(main_frame, text="Seller Type:", font=("Segoe UI", 12), bg="#f0f4fc")
selling_label.grid(row=4, column=0, sticky="w", pady=5)
selling_var = tk.StringVar(value="Dealer")
selling_menu = tk.OptionMenu(main_frame, selling_var, "Dealer", "Individual")
selling_menu.config(font=("Segoe UI", 12), width=18, bg="#228B22", fg="white", activebackground="#006400", activeforeground="white", relief="groove", highlightbackground="#006400")
selling_menu.grid(row=4, column=1, pady=5)

# Transmission
trans_label = tk.Label(main_frame, text="Transmission:", font=("Segoe UI", 12), bg="#f0f4fc")
trans_label.grid(row=5, column=0, sticky="w", pady=5)
trans_var = tk.StringVar(value="Manual")
trans_menu = tk.OptionMenu(main_frame, trans_var, "Manual", "Automatic")
trans_menu.config(font=("Segoe UI", 12), width=18, bg="#228B22", fg="white", activebackground="#006400", activeforeground="white", relief="groove", highlightbackground="#006400")
trans_menu.grid(row=5, column=1, pady=5)

# Owner
owner_label = tk.Label(main_frame, text="Number of Previous Owners (0-3):", font=("Segoe UI", 12), bg="#f0f4fc")
owner_label.grid(row=6, column=0, sticky="w", pady=5)
owner_entry = tk.Entry(main_frame, font=("Segoe UI", 12), width=20)
owner_entry.grid(row=6, column=1, pady=5)

# Predict Button


predict_btn = tk.Button(main_frame, text="Predict Price", command=on_predict, font=("Segoe UI", 13, "bold"), bg="#006400", fg="white", activebackground="#228B22", activeforeground="white", relief="flat", padx=10, pady=6)
predict_btn.grid(row=7, column=0, columnspan=2, pady=15)


# Result Label
result_label = tk.Label(main_frame, text="", font=("Segoe UI", 14, "bold"), fg="#006400", bg="#eaffea", wraplength=400, justify="center")
result_label.grid(row=8, column=0, columnspan=2, pady=10)

root.mainloop()
