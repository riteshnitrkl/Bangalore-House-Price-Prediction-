# train_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import joblib

# Load and clean your data
df = pd.read_csv('cleaned_bangalore_data.csv')

# Convert categorical location to one-hot encoding
dummies = pd.get_dummies(df['location'])
X = pd.concat([df[['total_sqft', 'bath', 'bhk']], dummies], axis=1)
y = df['price']

# Train the Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'ridgeModel.pkl')

print("✅ Model trained and saved as ridgeModel.pkl")
