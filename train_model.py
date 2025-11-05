import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

# Paths (assumes CSV at repo root)
CSV_PATH = os.path.join(os.path.dirname(__file__), "cleaned_bangalore_data.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "ridgeModel.pkl")

# Load data
df = pd.read_csv(CSV_PATH)

# Features and target
FEATURES = ['total_sqft', 'bath', 'bhk', 'location']
X = df[FEATURES].copy()
y = df['price']  # keep units the same as original (likely Lakhs)

# Preprocessing:
# - OneHotEncode 'location' (ignore unknown categories at inference time)
# - pass through numeric columns unchanged
categorical_features = ['location']
numeric_features = ['total_sqft', 'bath', 'bhk']

preprocessor = ColumnTransformer(
    transformers=[
        ("loc_ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features)
    ],
    remainder="passthrough"  # numeric features are kept as-is and appended after the encoded cols
)

# Pipeline: preprocessing -> ridge regressor
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("ridge", Ridge(alpha=1.0))
])

# Fit
pipeline.fit(X, y)

# Save pipeline
joblib.dump(pipeline, MODEL_OUT)

print(f"✅ Pipeline trained and saved to {MODEL_OUT}")
