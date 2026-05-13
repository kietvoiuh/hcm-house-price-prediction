import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

from preprocess import load_data


# =========================
# Load Data
# =========================
X, y = load_data(
    "data/processed/processed_house_data.csv"
)


# =========================
# Load Model
# =========================
pipeline = joblib.load(
    "models/best_model.pkl"
)


# =========================
# Get Preprocessor + Model
# =========================
preprocessor = pipeline.named_steps["preprocessor"]

model = pipeline.named_steps["model"]


# =========================
# Transform Features
# =========================
X_transformed = preprocessor.transform(X)


# =========================
# Feature Names
# =========================
feature_names = preprocessor.get_feature_names_out()


# =========================
# Feature Importance
# =========================
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(importance_df.head(15))


# =========================
# Plot Feature Importance
# =========================
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"].head(15)[::-1],
    importance_df["Importance"].head(15)[::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title("Top 15 Feature Importance")

plt.tight_layout()

plt.show()


# =========================
# SHAP
# =========================
explainer = shap.TreeExplainer(model)

# sample nhỏ để chạy nhanh
sample_size = min(300, X_transformed.shape[0])

X_sample = X_transformed[:sample_size]

shap_values = explainer.shap_values(X_sample)


# =========================
# SHAP Summary Plot
# =========================
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
)