import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.ensemble import RandomForestRegressor

from preprocess import load_data


# =========================
# Load Data
# =========================
X, y = load_data(r"D:\Tự Học\hcm-house-price-prediction\data\processed\processed_house_data.csv")


# =========================
# Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# =========================
# Column Types
# =========================
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


# =========================
# Preprocessing
# =========================
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features,
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# =========================
# Pipeline
# =========================
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
            ),
        ),
    ]
)


# =========================
# Hyperparameter Search
# =========================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}


# =========================
# KFold Cross Validation
# =========================
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# =========================
# GridSearchCV
# =========================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=kfold,
    scoring="r2",
    n_jobs=-1,
    verbose=2,
)


# =========================
# Train
# =========================
grid_search.fit(X_train, y_train)


# =========================
# Best Model
# =========================
best_model = grid_search.best_estimator_

print("\nBest Params:")
print(grid_search.best_params_)

print("\nBest CV Score:")
print(grid_search.best_score_)


# =========================
# Evaluate Test Set
# =========================
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n===== TEST METRICS =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")


# =========================
# Save Model
# =========================
joblib.dump(
    best_model,
    "models/best_model.pkl",
)

print("\nSaved best model!")