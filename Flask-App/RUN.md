Running the Flask app and using the CLI predictor
===============================================

This document explains how to run the Flask web app (`app.py`) and how to call the CLI `predict.py` provided in this folder.

Prerequisites
-------------
- Python 3.8+ installed
- Recommended: create a virtual environment and install required packages.

Quick setup (PowerShell)
------------------------
```powershell
# Go to the Flask app folder
cd "k:\My Drive\Projects\Personal\DS-Learn\ml-zoomcamp\midterm\Flask-App"

# Create and activate a virtual environment (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (adjust if you have a requirements.txt)
pip install --upgrade pip
pip install flask pandas numpy scikit-learn
```

Place the trained model
-----------------------
- The app expects the model file `extra_trees_energy_model.bin` to live in the same folder as `app.py`.
- After training (or when you obtain the trained bundle), copy it here:
```powershell
copy "..\extra_trees_energy_model.bin" .
# or move the file into this folder in Explorer
```

Run the Flask app (local)
-------------------------
```powershell
# Start the app (development server)
python app.py
# The app will start on http://0.0.0.0:5000 (open http://localhost:5000 in the browser)
```

Use the web UI
--------------
- Open `http://localhost:5000` in your browser. The page shows a preview of model features and a JSON form to send a single prediction.

Call the API (example curl)
---------------------------
- Single sample (JSON object of feature:value pairs):
```powershell
curl -X POST -H "Content-Type: application/json" -d "{\"T1\": 20.5, \"RH_1\": 30, \"Hour_sin\": 0.0, ... }" http://localhost:5000/predict
```

- The response is JSON: `{ "predictions": [ <value> ], "n_features": <int> }`.

Using `predict.py` from the command line
--------------------------------------
- `predict.py` supports single-dict prediction and batch CSV prediction. Example usages from this folder:

Single JSON (print result):
```powershell
python predict.py -d "{\"T1\":20.5, \"RH_1\":30, \"Hour_sin\":0.0, ... }"
```

Batch CSV (output printed):
```powershell
python predict.py -b path\to\my_inputs.csv
```

Notes and troubleshooting
-------------------------
- The feature names used by the model are recorded inside the model bundle under the `features` key. Ensure any input JSON or CSV includes these columns (or at least the keys) — missing features will be filled with zeros by the app.
- If you changed the model filename or structure, update `MODEL_PATH` inside `app.py` or place the model file in this directory with the expected name.
- For production use, run the Flask app behind a WSGI server (Gunicorn, Waitress) and disable `debug=True`.

If you want, I can also:
- Add a `requirements.txt` for exact package versions.
- Add example input CSVs and a small `preprocess.py` to transform raw data into model features.

Enjoy! 