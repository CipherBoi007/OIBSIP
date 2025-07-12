import tkinter as tk
from tkinter import messagebox
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing function (same as in training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

# Prediction function
def predict_spam(message):
    cleaned = preprocess_text(message)
    features = tfidf.transform([cleaned])
    prediction = model.predict(features)[0]
    pred_text = 'SPAM' if prediction == 1 else 'HAM'
    return pred_text

# GUI setup
def on_predict():
    user_input = entry.get()
    if not user_input.strip():
        messagebox.showwarning("Input Error", "Please enter a message.")
        return
    pred_text = predict_spam(user_input)
    if pred_text == 'SPAM':
        prediction_label.config(text=f"{pred_text}", fg="#d32f2f")
    else:
        prediction_label.config(text=f"{pred_text}", fg="#388e3c")
    confidence_label.config(text="")

root = tk.Tk()
root.title("Spam Detector - Modern GUI")
root.geometry("450x320")
root.configure(bg="#f0f4fc")

# Header
header = tk.Label(root, text="📧 Spam Detector", font=("Segoe UI", 20, "bold"), bg="#4f8ef7", fg="white", pady=12)
header.pack(fill="x")

# Main frame for padding
main_frame = tk.Frame(root, bg="#f0f4fc")
main_frame.pack(expand=True, fill="both", padx=20, pady=10)

label = tk.Label(main_frame, text="Enter your message:", font=("Segoe UI", 13), bg="#f0f4fc", fg="#333")
label.pack(pady=(10, 5))

entry = tk.Entry(main_frame, width=44, font=("Segoe UI", 12), relief="solid", bd=2)
entry.pack(pady=5, ipady=4)

predict_btn = tk.Button(main_frame, text="Predict", command=on_predict, font=("Segoe UI", 12, "bold"), bg="#4f8ef7", fg="white", activebackground="#357ae8", activeforeground="white", relief="flat", padx=10, pady=6)
predict_btn.pack(pady=15)


# Output labels for prediction and confidence
prediction_label = tk.Label(main_frame, text="", font=("Segoe UI", 15, "bold"), bg="#f0f4fc", wraplength=380, justify="center")
prediction_label.pack(pady=(10, 2))

confidence_label = tk.Label(main_frame, text="", font=("Segoe UI", 12), bg="#f0f4fc", wraplength=380, justify="center")
confidence_label.pack(pady=(2, 10))

root.mainloop()
