from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = load_model("attrition_model.h5")
scaler = joblib.load("scaler.pkl")  # Save this after training

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    
    # One-hot encode and scale
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_input_columns, fill_value=0)
    X = scaler.transform(df)
    
    prediction = model.predict(X)[0][0]
    result = "Likely to Leave" if prediction > 0.5 else "Likely to Stay"
    
    return jsonify({"prediction": result, "probability": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)