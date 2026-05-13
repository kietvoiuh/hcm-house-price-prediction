import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(r"D:\Tự Học\hcm-house-price-prediction\data\processed\data_cleaned_colab.csv")


# =========================
# CLEAN DATA
# =========================

# Xóa dữ liệu null

df.dropna(inplace=True)


# =========================
# FIX DISTRICT
# =========================

# Tạm thời extract district từ Full_Info
# Vì cột District hiện đang lỗi

DISTRICTS = [
    "Quận 1",
    "Quận 2",
    "Quận 3",
    "Quận 4",
    "Quận 5",
    "Quận 6",
    "Quận 7",
    "Quận 8",
    "Quận 9",
    "Quận 10",
    "Quận 11",
    "Quận 12",
    "Bình Thạnh",
    "Gò Vấp",
    "Tân Bình",
    "Tân Phú",
    "Phú Nhuận",
    "Thủ Đức",
    "Bình Chánh",
    "Nhà Bè"
]


def extract_district(text):
    text = str(text)

    for d in DISTRICTS:
        if d.lower() in text.lower():
            return d

    return "Unknown"


# Tạo district mới

df["District"] = df["Full_Info"].apply(extract_district)



# =========================
# FEATURE ENGINEERING
# =========================

# Giá trên mét vuông

df["Price_per_m2"] = df["Price_Million"] / df["Area_m2"]

# Log diện tích

df["Area_log"] = np.log1p(df["Area_m2"])


# Nhóm diện tích

def area_category(area):
    if area < 50:
        return "small"
    elif area < 100:
        return "medium"
    else:
        return "large"



df["Area_category"] = df["Area_m2"].apply(area_category)


# =========================
# DISTANCE FEATURE
# =========================

DISTANCE_MAP = {
    "Quận 1": 0,
    "Quận 3": 2,
    "Quận 4": 3,
    "Quận 7": 7,
    "Bình Thạnh": 4,
    "Thủ Đức": 10,
    "Bình Chánh": 15,
    "Unknown": 20
}


df["Distance_To_D1"] = df["District"].map(DISTANCE_MAP)


df["Distance_To_D1"] = df["Distance_To_D1"].fillna(12)


# =========================
# POPULATION DENSITY
# =========================

POPULATION_DENSITY = {
    "Quận 1": 26000,
    "Quận 3": 39000,
    "Quận 7": 9000,
    "Bình Thạnh": 24000,
    "Thủ Đức": 12000,
    "Unknown": 10000
}


df["Population_Density"] = df["District"].map(POPULATION_DENSITY)


df["Population_Density"] = df[
    "Population_Density"
].fillna(10000)


# =========================
# ENCODE CATEGORICAL
# =========================

categorical_cols = [
    "District",
    "Area_category"
]


df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)


# =========================
# STANDARDIZE FEATURES
# =========================

numerical_cols = [
    "Area_m2",
    "Price_per_m2",
    "Area_log",
    "Distance_To_D1",
    "Population_Density"
]


scaler = StandardScaler()


df[numerical_cols] = scaler.fit_transform(
    df[numerical_cols]
)


# =========================
# SAVE PROCESSED DATA
# =========================

output_path = r"D:\Tự Học\hcm-house-price-prediction\data\processed\processed_house_data.csv"


df.to_csv(output_path, index=False)


print("Preprocessing completed!")
print(df.head())


# =========================
# Update
# =========================

def load_data(file_path):
    df = pd.read_csv(file_path)

    target_col = 'Price_Million'
    
    cols_to_drop = [target_col, 'Full_Info', 'Price_per_m2']

    existing_drops = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=existing_drops)
    y = df[target_col]
    
    return X, y