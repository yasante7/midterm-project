"""
CLI helper to load the saved model bundle and run a sample prediction.
Usage:
  python predict.py --json '{"T1": 20, "RH_1": 40, ...}'

It expects `et_model.bin` in the same folder.
"""
import argparse
import json
import os
import pickle
import pandas as pd


def load_bundle(path='et_model.bin'):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model bundle not found at {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_from_json(bundle, json_str):
    data = json.loads(json_str)
    # Accept top-level dict representing feature->value
    features = bundle.get('features')
    scaler = bundle.get('scaler')
    model = bundle.get('model')

    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)
    X = df.values.astype(float)
    if scaler is not None:
        X = scaler.transform(X)
    preds = model.predict(X)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, help='JSON string with feature:value pairs')
    args = parser.parse_args()
    bundle = load_bundle()
    if not args.json:
        print('Provide --json with a single-sample feature mapping. Example:')
        print(json.dumps({k: 0 for k in bundle.get('features')[:10]}, indent=2))
        return
    preds = predict_from_json(bundle, args.json)
    print('Predictions:', preds)


if __name__ == '__main__':
    main()
