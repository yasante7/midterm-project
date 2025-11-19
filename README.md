# Appliances Energy Prediction — Project README

## Project Summary

This project develops and evaluates machine learning models to predict household appliance energy consumption (Wh) using the UCI "Appliances Energy Prediction" dataset. The analysis includes extensive feature engineering, model training and hyperparameter tuning, model comparison, and a lightweight model-serving prototype (Flask).

Key outcome: ensemble tree models performed best — the Extra Trees model achieved a test RMSE ≈ 63.07 and R² ≈ 0.60 (reported in the notebook analyses).

## Problem Statement

Predict short-term appliance energy consumption in a single-family residential building using environmental (indoor/outdoor) measurements and engineered temporal features. Accurate predictions can support demand-side management, energy-efficiency interventions, and intelligent control systems.

## Dataset

- Source: UCI Machine Learning Repository — "Appliances Energy Prediction" dataset.
- Instances: ~19,735 measurements taken every 10 minutes.
- Key variables: `Appliances` (target, Wh), room temperatures `T1..T9`, humidities `RH_*`, outdoor weather features (`T_out`, `RH_out`, `Windspeed`, `Visibility`, `Press_mm_hg`, `Tdewpoint`), `lights`, and two random variables `rv1`, `rv2`.

Data is stored in `data/energydata_complete.csv` and is preprocessed, feature-engineered and split in `data_analysis.ipynb`.

## Feature Engineering & EDA (brief)

- Temporal features extracted from the timestamp: seconds-from-midnight (NSM), hour, day, month, day-of-week, week status (Weekend/Weekday), and cyclic encodings for hour (`Hour_sin`, `Hour_cos`).
- Seasonal categorical variable derived from month (Winter, Spring, Summer, Autumn).
- Basic exploratory plots produced: pairplots, time series profiles, histograms, boxplots, and heatmaps to inspect hourly/weekly consumption patterns.
- Correlation heatmap computed for numeric columns to inspect multicollinearity.

## Modelling Approach

Models trained and compared:
- Multiple Linear Regression (LM)
- Logistic Regression (Logit) — included as baseline (note: logistic is not ideal for regression targets; kept for experimentation)
- Decision Tree Regressor (DT)
- Random Forest Regressor (RF)
- Extra Trees Regressor (ET)
- Gradient Boosting Regressor (GB)
- XGBoost (XGB)
- Support Vector Regression (SVR)

Model selection & tuning
- Data splits: training / validation / test splits created in the notebook.
- Preprocessing: one-hot encoding of categorical variables (day-of-week, season etc.) and `StandardScaler` for numeric features.
- Hyperparameter tuning performed using `HalvingRandomSearchCV` / `HalvingGridSearchCV` (efficient halving search) for tree and boosting models.
- Evaluation via cross-validation and hold-out test set.

Evaluation metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²)

Key result (notebook)
- The Extra Trees model (ET) performed best overall on these experiments with test RMSE ≈ 63.07 and R² ≈ 0.60. Random Forest was close behind.

## Project Files (important)

- `data_analysis.ipynb` — full analysis, EDA, feature engineering, training, hyperparameter tuning and notebook cells that save artifacts.
- `data/energydata_complete.csv` — original dataset.
- `new_data/` — CSVs with training/validation/test splits and saved arrays created by the notebook (`X_train.csv`, `y_train.csv`, etc.).
- `models/` — serialized model artifacts saved by the notebook (e.g. `models/best_et.pkl`).
- `et_model.bin` — a bundled pickle file (created by the notebook) containing `{'model': estimator, 'scaler': scaler, 'features': features}` used by the serving app.
- `app.py` — simple Flask app that loads `et_model.bin` and serves predictions via `POST /predict` and a minimal web UI.
- `predict.py` — small CLI helper to run a single-sample prediction using `et_model.bin`.
- `requirements.txt` — Python dependency list.

## How to run locally (PowerShell)

1) (Optional) Create and activate a virtual environment

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Ensure the notebook has created the model bundle `et_model.bin` in this folder. If not, run the final notebook cell that creates `model_bundle` and pickles it.

4) Start the Flask app

```powershell
python app.py
```

The server listens on `http://0.0.0.0:5000/` by default. Use the web UI or send JSON to `/predict`.

## API Usage Examples

POST a single-sample feature dict (PowerShell `curl`):

```powershell
curl -X POST -H "Content-Type: application/json" -d '{"T1": 20, "RH_1": 40}' http://127.0.0.1:5000/predict
```

The endpoint accepts either a top-level feature mapping or an object with a `features` key, e.g. `{"features": {"T1": 20, ...}}`.

Using the CLI helper:

```powershell
python predict.py --json '{"T1": 20, "RH_1": 40}'
```

## Docker (example)

Create a `Dockerfile` like the following and build the image. This example is a recommended convenience; a `Dockerfile` is not included by default in this folder but may be added for submission.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

Build & run (local):

```bash
docker build -t appliances-energy-app .
docker run -p 5000:5000 appliances-energy-app
```

## Limitations & Next Steps

- Dataset scope: the UCI dataset is from a single household/location and may not generalise to other homes or climates.
- Model bundle: the current serving approach pickles a model+scaler+feature list. For production, prefer a reproducible pipeline (e.g. sklearn Pipeline or joblib) and avoid insecure pickle loading for untrusted inputs.
- Preprocessing gaps: the server assumes one-hot encoding and numeric inputs; it fills missing features with zeros. A more robust preprocessor that replicates notebook encoding (categories, missing-value handling) would improve reliability.
- Deployment: add integration tests, CI checks, a Dockerfile in the repo, and optionally deploy to a cloud platform.
- Monitoring & fairness: add drift detection and monitoring for model degradation over time.

## Reproducibility

To reproduce the results, run `data_analysis.ipynb` end-to-end (or refactor to `train.py`) to re-create splits, train models and save `et_model.bin`. The notebook contains the exact preprocessing and hyperparameter search steps used in the experiments.

## References

Include paper and dataset references in your final report. Use citation placeholders like (Author, Year) if needed.
