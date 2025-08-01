from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model and dataset
model = joblib.load('ridgeModel.pkl')
df = pd.read_csv('cleaned_bangalore_data.csv')

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# API to get unique locations
@app.route('/get_location_names')
def get_location_names():
    locations = sorted(df['location'].unique())
    return jsonify({'locations': locations})

# Dummy handlers for now (you can remove these if you’re not using 'area' or 'availability')
@app.route('/get_area_names')
def get_area_names():
    return jsonify({'area': ['Any']})

@app.route('/get_availability_names')
def get_availability_names():
    return jsonify({'availability': ['Ready to move', 'Under Construction']})

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    location = request.form.get('loc')
    sqft = float(request.form.get('sqft'))
    bath = float(request.form.get('bath'))
    bhk = float(request.form.get('bhk'))

    # For simplicity, we'll do one-hot encoding manually (minimal setup)
    # Create a zero array with length = number of locations
    locations = sorted(df['location'].unique())
    x = np.zeros(len(locations) + 3)  # 3 = sqft, bath, bhk

    # Set values
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in locations:
        loc_index = locations.index(location)
        x[3 + loc_index] = 1

    # Predict using ridge model
    price = model.predict([x])[0]

    return render_template('index.html', prediction_text=f"Estimated Price: ₹ {round(price, 2)} Lakhs")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
