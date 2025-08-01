
# Bangalore House Price Prediction

A machine learning web application that predicts house prices in Bangalore based on key features such as location, square footage, number of BHKs, and bathrooms. Built with Python, Flask, and Ridge Regression.

---

## 🔧 Tech Stack

- **Frontend**: HTML + CSS
- **Backend**: Flask (Python)
- **ML Model**: Ridge Regression (scikit-learn)
- **Other Tools**: Pandas, NumPy, Seaborn, Matplotlib, Joblib

---

## 🚀 How It Works

1. User inputs data (location, sqft, BHK, bathrooms)
2. Flask sends the data to the Ridge regression model
3. Model returns the predicted price 💰
4. Output is displayed on the web interface

---

python3 -m venv venv
source venv/bin/activate  


Install dependencies



pip install flask pandas numpy scikit-learn matplotlib seaborn joblib


Run the app
python3 main.py
Visit http://127.0.0.1:5000 in your browser.



