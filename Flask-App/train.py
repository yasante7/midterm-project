#!/usr/bin/env python
# coding: utf-8

import pickle
import warnings
warnings.filterwarnings('ignore')

# Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error

# -----------------------------
# Parameters
# -----------------------------
n_splits = 10
output_file = "extra_trees_energy_model.bin"


# -----------------------------
# Load and Prepare Data
# -----------------------------
energy_data = pd.read_csv("data/energydata_complete.csv")

energy_data['date'] = pd.to_datetime(energy_data['date'], format="%Y-%m-%d %H:%M:%S")

# -----------------------------
# Feature Engineering
# -----------------------------
def second_day(x):
    return x.hour * 3600 + x.minute * 60 + x.second

def weekend_weekday(x):
    return 'Weekend' if x.weekday() >= 5 else 'Weekday'

def get_season(month):
    if month in [3,4,5]: return 'Spring'
    if month in [6,7,8]: return 'Summer'
    if month in [9,10,11]: return 'Autumn'
    return 'Winter'

energy_data['NSM'] = energy_data['date'].apply(second_day)
energy_data['WeekStatus'] = energy_data['date'].apply(weekend_weekday)
energy_data['Day_of_week'] = energy_data['date'].dt.day_name()
energy_data['Hour'] = energy_data['date'].dt.hour
energy_data['Month'] = energy_data['date'].dt.month
energy_data['Day'] = energy_data['date'].dt.day
energy_data['Season'] = energy_data['Month'].apply(get_season)

# Cyclical encoding
energy_data['Hour_sin'] = np.sin(2 * np.pi * energy_data['Hour'] / 24)
energy_data['Hour_cos'] = np.cos(2 * np.pi * energy_data['Hour'] / 24)

# Convert categorical variables
cat_cols = ['Day_of_week', 'WeekStatus', 'Season']
for c in cat_cols:
    energy_data[c] = energy_data[c].astype('category')

data = energy_data.copy()

# -----------------------------
# Train–Val–Test Split
# -----------------------------
train_data, val_test_data = train_test_split(data, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

# One-hot encoding
categorical_cols = data.select_dtypes('category').columns.tolist()

train = pd.get_dummies(train_data, columns=categorical_cols)
val   = pd.get_dummies(val_data, columns=categorical_cols)
test  = pd.get_dummies(test_data, columns=categorical_cols)

# Convert booleans
for df in (train, val, test):
    df[df.select_dtypes('bool').columns] = df.select_dtypes('bool').astype(int)

# Align columns
train, val = train.align(val, join="left", axis=1, fill_value=0)
train, test = train.align(test, join="left", axis=1, fill_value=0)

# -----------------------------
# Prepare Features
# -----------------------------
target = 'Appliances'
features = [col for col in train.columns if col not in [target, 'date']]

scaler = StandardScaler()

X_train = scaler.fit_transform(train[features])
y_train = train[target]

X_val = scaler.transform(val[features])
y_val = val[target]

X_test = scaler.transform(test[features])
y_test = test[target]


# -----------------------------
# ExtraTrees Regressor
# -----------------------------
print("Training ExtraTreesRegressor...")
et = ExtraTreesRegressor(random_state=1)
et.fit(X_train, y_train)

# -----------------------------
# Cross-validation
# -----------------------------
print("Performing Cross-Validation...")
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
cv_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_vl = X_train[train_idx], X_train[val_idx]
    y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = ExtraTreesRegressor(random_state=1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_vl)

    rmse = float(np.sqrt(mean_squared_error(y_vl, y_pred)))
    cv_results.append({'Fold': fold, 'RMSE': rmse})

cv_summary = pd.DataFrame(cv_results)
print("\nCross-Validation Summary:\n")
print(cv_summary)
print("\nMean RMSE:", cv_summary['RMSE'].mean())
print("Std RMSE:", cv_summary['RMSE'].std())


# -----------------------------
# Evaluate Model
# -----------------------------
def evaluate(model):
    preds = model.predict(X_test)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "MAE": mean_absolute_error(y_test, preds),
        "MAPE": mean_absolute_percentage_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

final_metrics = evaluate(et)
print("\nFinal Test Metrics:\n", final_metrics)

# -----------------------------
# Save Model + Scaler
# -----------------------------
with open(output_file, "wb") as f_out:
    # Save explicit dict bundle for clarity and forward compatibility
    bundle = {"model": et, "scaler": scaler, "features": features}
    pickle.dump(bundle, f_out)

print(f"\nModel saved as: {output_file}")