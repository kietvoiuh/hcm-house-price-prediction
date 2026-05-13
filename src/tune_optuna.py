import optuna
import joblib

from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    KFold,
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from preprocess import load_data

X, y = load_data(
    "D:\Tự Học\hcm-house-price-prediction\data\processed\processed_house_data.csv"
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


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
# Objective Function
# =========================
def objective(trial):

    model = RandomForestRegressor(
        n_estimators=trial.suggest_int(
            "n_estimators",
            100,
            500,
        ),
        max_depth=trial.suggest_int(
            "max_depth",
            5,
            30,
        ),
        min_samples_split=trial.suggest_int(
            "min_samples_split",
            2,
            10,
        ),
        min_samples_leaf=trial.suggest_int(
            "min_samples_leaf",
            1,
            5,
        ),
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring="r2",
        n_jobs=-1,
    )

    return scores.mean()


# =========================
# Run Study
# =========================
study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=30,
)


print("Best Trial:")
print(study.best_trial.params)
print(study.best_value)


# =========================
# Train Final Model
# =========================
best_model = RandomForestRegressor(
    **study.best_trial.params,
    random_state=42,
    n_jobs=-1,
)

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            best_model,
        ),
    ]
)

final_pipeline.fit(X_train, y_train)


y_pred = final_pipeline.predict(X_test)

print("\nTest R2:")
print(r2_score(y_test, y_pred))


joblib.dump(
    final_pipeline,
    "models/best_model.pkl",
)