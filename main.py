from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Robust paths (works on Render)
MODEL_PATH = os.path.join(app.root_path, "ridgeModel.pkl")
CSV_PATH = os.path.join(app.root_path, "cleaned_bangalore_data.csv")

# Load pipeline/model and dataset
model = joblib.load(MODEL_PATH)  # saved Pipeline (preprocessor + estimator)
df = pd.read_csv(CSV_PATH)

# UI lists
LOCATIONS_UI = sorted(df['location'].unique())

@app.route('/')
def index():
    return render_template('index.html', prediction_text="")

@app.route('/get_location_names')
def get_location_names():
    return jsonify({'locations': LOCATIONS_UI})

@app.route('/get_area_names')
def get_area_names():
    return jsonify({'area': ['Any']})

@app.route('/get_availability_names')
def get_availability_names():
    return jsonify({'availability': ['Ready to move', 'Under Construction']})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # read inputs
        location = request.form.get('loc', '').strip()
        sqft_raw = request.form.get('sqft', '').strip()
        bath_raw = request.form.get('bath', '').strip()
        bhk_raw = request.form.get('bhk', '').strip()

        # validate
        if not sqft_raw or not bath_raw or not bhk_raw or not location:
            raise ValueError("Please provide values for location, total_sqft, bath and bhk.")

        total_sqft = float(sqft_raw)
        bath = int(float(bath_raw))
        bhk = int(float(bhk_raw))

        # Build a single-row DataFrame with the same raw columns used in training
        X_input = pd.DataFrame([{
            'total_sqft': total_sqft,
            'bath': bath,
            'bhk': bhk,
            'location': location
        }])

        # model is a pipeline: it will perform encoding exactly as training
        pred = model.predict(X_input)[0]

        # Format result. Your training target 'price' appears to be in Lakhs (keep consistent)
        pred_text = f"Estimated Price: ₹ {round(pred, 2)} Lakhs"

    except Exception as e:
        app.logger.exception("Prediction error")
        pred_text = f"Error making prediction: {e}"

    return render_template('index.html', prediction_text=pred_text)

# In production Gunicorn will import the app
if __name__ == "__main__":
    app.run(port=5001)
