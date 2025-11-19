#!/usr/bin/env python3
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
import os
from flask import Flask, request, jsonify


# -------------------------
# Load Model + Scaler + Features
# -------------------------
MODEL_PATH = os.environ.get('MODEL_PATH', 'extra_trees_energy_model.bin')

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f'Model file not found: {MODEL_PATH}')

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

# Support dict bundle or tuple
if isinstance(bundle, dict):
    model = bundle.get('model')
    scaler = bundle.get('scaler')
    features = bundle.get('features')
else:
    model, scaler, features = bundle

print(f"Loaded model with {len(features)} features.")


# -------------------------
# Flask App
# -------------------------
app = Flask('energy_predictor')


# -------------------------
# Prediction helper
# -------------------------
def prepare_df_for_prediction(df_raw):
    # One-hot encode categorical variables present in df_raw
    df = pd.get_dummies(df_raw)

    # Ensure all expected training features exist
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Keep feature order
    df = df[features]
    return df


def predict_batch(df_raw):
    df = prepare_df_for_prediction(df_raw)
    if scaler is not None:
        X = scaler.transform(df)
    else:
        X = df.values
    preds = model.predict(X)
    return preds


# -------------------------
# Endpoints
# -------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'}), 200


@app.route('/predict', methods=['POST'])
def predict_api():
    payload = request.get_json(force=True)
    if payload is None:
        return jsonify({'error': 'No JSON payload provided'}), 400

    # Accept either a list of records or an object with key 'data'
    records = payload.get('data') if isinstance(payload, dict) and 'data' in payload else payload
    if records is None:
        return jsonify({'error': 'JSON must be a list of records or an object with key `data`'}), 400

    try:
        df = pd.DataFrame(records)
        preds = predict_batch(df)
        return jsonify({'predictions': preds.tolist()}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9696))
    print(f'Starting server on http://0.0.0.0:{port} ...')
    app.run(host='0.0.0.0', port=port)
