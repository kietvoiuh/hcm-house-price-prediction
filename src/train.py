import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor



# =========================
# LOAD DATA
# =========================

file_path = r"D:\Tự Học\hcm-house-price-prediction\data\processed\processed_house_data.csv"


df = pd.read_csv(file_path)


# =========================
# FEATURES & TARGET
# =========================

X = df.drop([
    "Price_Million",
    "Full_Info"
], axis=1)


y = df["Price_Million"]



# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# MODEL LIST
# =========================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}


# =========================
# TRAIN & EVALUATE
# =========================

results = []

best_model = None
best_r2 = -999
best_model_name = ""


for name, model in models.items():

    print(f"========== {name} ==========")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(y_test, y_pred)

    # Save result
    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Save best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name


# =========================
# RESULTS TABLE
# =========================

results_df = pd.DataFrame(results)

print("==============================")
print("MODEL COMPARISON")
print("==============================")
print(results_df)


# =========================
# SAVE BEST MODEL
# =========================

import os

os.makedirs("models", exist_ok=True)

model_path = "models/best_model.pkl"

joblib.dump(best_model, model_path)

print(f"\nBest Model: {best_model_name}")
print(f"Saved to: {model_path}")