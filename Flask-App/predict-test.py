import pickle
import pandas as pd

# Load the model bundle you saved in train.py
with open("extra_trees_energy_model.bin", "rb") as f:
    model, scaler, features = pickle.load(f)

# Your input sample:
sample = {
 'lights': 10,
 'T1': 21.5,
 'RH_1': 36.6633333333333,
 'T2': 18.73,
 'RH_2': 39.223333333333294,
 'T3': 22.9266666666667,
 'RH_3': 34.4,
 'T4': 21.7,
 'RH_4': 34.3214285714286,
 'T5': 20.9266666666667,
 'RH_5': 65.8933333333333,
 'T6': 5.29666666666667,
 'RH_6': 53.55666666666671,
 'T7': 21.40375,
 'RH_7': 32.53375,
 'T8': 22.79,
 'RH_8': 38.7,
 'T9': 20.463333333333296,
 'RH_9': 36.76,
 'T_out': 4.15,
 'Press_mm_hg': 758.0666666666668,
 'RH_out': 87.1666666666667,
 'Windspeed': 4.0,
 'Visibility': 26.5,
 'Tdewpoint': 2.2,
 'rv1': 16.421150916721672,
 'rv2': 16.421150916721672,
 'NSM': 33000,
 'WeekStatus': 'Weekend',
 'Day_of_week': 'Sunday',
 'Hour': 9,
 'Month': 4,
 'Day': 24,
 'Season': 'Spring',
 'Hour_sin': 0.7071067811865476,
 'Hour_cos': -0.7071067811865475
}

# Put into a dataframe
df = pd.DataFrame([sample])

# One-hot encode for the categorical features
df = pd.get_dummies(df)

# Ensure same column order as training
for col in features:
    if col not in df.columns:
        df[col] = 0

df = df[features]

# Scale
X = scaler.transform(df)

# Predict
pred = model.predict(X)[0]
print("Predicted appliance energy use:", pred)