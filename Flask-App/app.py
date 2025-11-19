import os
import pickle
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, 'extra_trees_energy_model.bin')

app = Flask(__name__)

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found at: {path}')
    with open(path, 'rb') as f:
        model_bundle = pickle.load(f)
    # Expecting dict with keys: model, scaler, features
    return model_bundle

MODEL_BUNDLE = load_model()


INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Appliances Energy - Prediction</title>
    <style>body{font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:2rem auto;padding:1rem}textarea{width:100%;height:220px}</style>
  </head>
  <body>
    <h2>Appliances Energy Prediction</h2>
    <p>Submit a JSON payload with either:</p>
    <ul>
      <li>a single object containing feature:value pairs (keys matching model features), or</li>
      <li>an object with key <code>features</code> pointing to that object, e.g. <code>{"features": {"T1": 20, ...}}</code></li>
    </ul>
    <form id="jsonForm" method="post" action="/predict">
      <label for="payload">JSON payload</label>
      <textarea id="payload" name="payload">{}</textarea>
      <br/>
      <button type="submit">Send</button>
    </form>
    <hr/>
    <h3>Model Features (partial preview)</h3>
    <pre id="features">Loading features...</pre>
    <script>
      fetch('/_features').then(r=>r.json()).then(d=>{
        document.getElementById('features').textContent = JSON.stringify(d.features, null, 2);
        document.getElementById('payload').textContent = JSON.stringify(Object.fromEntries(d.features.slice(0,50).map(f=>[f,0])), null, 2);
      });
      // Make the form submit JSON instead of form-encoded
      document.getElementById('jsonForm').addEventListener('submit', async function(e){
        e.preventDefault();
        let payloadText = document.getElementById('payload').value;
        let payload;
        try{ payload = JSON.parse(payloadText); } catch(err){ alert('Invalid JSON'); return; }
        const res = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body: JSON.stringify(payload)});
        const json = await res.json();
        alert(JSON.stringify(json, null, 2));
      });
    </script>
  </body>
</html>
"""


@app.route('/_features')
def _features():
    features = MODEL_BUNDLE.get('features') if MODEL_BUNDLE else []
    return jsonify({'features': features})


@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)


def prepare_input(df: pd.DataFrame, features):
    # Reindex to expected model features, filling missing with 0
    df = df.reindex(columns=features, fill_value=0)
    return df


@app.route('/predict', methods=['POST'])
def predict():
    if MODEL_BUNDLE is None:
        return jsonify({'error': 'Model not loaded'}), 500

    payload = None
    if request.is_json:
        payload = request.get_json()
    else:
        # support simple form post
        txt = request.form.get('payload') or request.data.decode('utf-8')
        try:
            import json
            payload = json.loads(txt) if txt else {}
        except Exception:
            return jsonify({'error': 'Request must be JSON or include a JSON payload in `payload`'}), 400

    # Accept different payload shapes
    if isinstance(payload, dict) and 'features' in payload and isinstance(payload['features'], dict):
        data_obj = payload['features']
    elif isinstance(payload, dict) and all(isinstance(v, (int, float, str)) for v in payload.values()):
        # treat top-level dict as feature mapping
        data_obj = payload
    else:
        # try list or array
        return jsonify({'error': 'Unsupported payload shape. Provide a feature:value dict or {"features": {...}}'}), 400

    features = MODEL_BUNDLE.get('features')
    scaler = MODEL_BUNDLE.get('scaler')
    model = MODEL_BUNDLE.get('model')

    # Build DataFrame from single sample
    df = pd.DataFrame([data_obj])
    df_prepared = prepare_input(df, features)

    X = df_prepared.values.astype(float)
    if scaler is not None:
        X = scaler.transform(X)

    preds = model.predict(X)
    preds_list = [float(p) for p in np.ravel(preds)]

    return jsonify({'predictions': preds_list, 'n_features': len(features)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
