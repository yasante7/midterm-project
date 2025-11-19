#!/usr/bin/env python3
"""
predict.py

Supports batch prediction (from DataFrame) and single prediction (from dict)
using the Extra Trees energy model, with optional CLI input.
"""

import os
import pickle
import pandas as pd
import numpy as np
import argparse
import json

DEBUG = True

# ======== CONFIG ========
MODEL_PATH = 'flask-app/extra_trees_energy_model.bin'
TARGET = 'Appliances'  # example target name, adjust if needed

# ======== LOAD MODEL ========
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

model = bundle.get('model')
scaler = bundle.get('scaler')
features = bundle.get('features')

if model is None or features is None:
    raise RuntimeError("Model bundle must contain 'model' and 'features'.")


# ======== HELPERS ========
def prepare_input(df: pd.DataFrame):
    """Align DataFrame columns to model features, fill missing with 0."""
    df = df.reindex(columns=features, fill_value=0)
    return df


# ======== BATCH PREDICTION ========
def predict_df(df: pd.DataFrame, verbose=DEBUG):
    """Predict on a batch DataFrame."""
    df_prepared = prepare_input(df)
    X = df_prepared.values.astype(float)
    if scaler:
        X = scaler.transform(X)
    y_pred = model.predict(X)
    if verbose:
        print(f"\nPredictions (first 10): {list(y_pred)[:10]}")
    return y_pred


# ======== SINGLE DICT PREDICTION ========
def predict_dict(input_dict: dict, verbose=DEBUG):
    """Predict for a single record (dict)."""
    df = pd.DataFrame([input_dict])
    df_prepared = prepare_input(df)
    X = df_prepared.values.astype(float)
    if scaler:
        X = scaler.transform(X)
    y_pred = model.predict(X)
    result = {TARGET.lower(): float(y_pred[0])}
    if verbose:
        print(f"Input: {input_dict}\nPrediction: {result}")
    return result


# ======== CLI ENTRY POINT ========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Appliances energy.')
    parser.add_argument('-d', '--dict', type=str, help='Single input as JSON string')
    parser.add_argument('-b', '--batch', type=str, help='CSV file for batch prediction')
    parser.add_argument('-df', '--dict-file', type=str, help='Path to JSON file for single prediction')
    args = parser.parse_args()

    if args.dict:
        try:
            input_dict = json.loads(args.dict)
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)
            exit(1)
        predict_dict(input_dict, verbose=True)

    elif args.batch:
        if not os.path.exists(args.batch):
            print("CSV file not found:", args.batch)
            exit(1)
        df = pd.read_csv(args.batch)
        predict_df(df, verbose=True)
        
    elif args.dict_file:
        import json
        with open(args.dict_file, 'r') as f:
            input_dict = json.load(f)
        predict_dict(input_dict, verbose=True)

    else:
        # Run quick test
        print("No input provided. Running built-in test.")
        batch_example = pd.DataFrame([
            {f: 0 for f in features},
            {f: 1 for f in features},
            {f: 0.5 for f in features}
        ])
        predict_df(batch_example, verbose=True)

        example_input = {f: 1 for f in features}
        predict_dict(example_input, verbose=True)
