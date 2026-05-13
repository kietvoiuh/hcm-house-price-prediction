# 🏠 HCM House Price Prediction

Dự án dự đoán giá nhà tại **TP. Hồ Chí Minh** sử dụng Machine Learning. Pipeline hoàn chỉnh từ thu thập dữ liệu, huấn luyện mô hình, đến triển khai REST API bằng FastAPI.

---

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [API Reference](#api-reference)
- [Mô hình & Kết quả](#mô-hình--kết-quả)
- [Docker](#docker)

---

## Tổng quan

Dự án xây dựng một hệ thống dự đoán giá nhà tại TP.HCM, bao gồm:

- Thu thập và tiền xử lý dữ liệu bất động sản thực tế
- So sánh và lựa chọn mô hình tốt nhất trong số: Linear Regression, Random Forest, XGBoost
- Tự động lưu model có R² cao nhất bằng `joblib`
- Triển khai REST API bằng FastAPI nhận JSON input và trả về giá dự đoán

---

## Cấu trúc dự án

```
hcm-house-price-prediction/
├── api/
│   └── main.py              # FastAPI app — REST API endpoint
├── data/
│   ├── raw/                 # Dữ liệu thô
│   └── processed/           # Dữ liệu đã xử lý (processed_house_data.csv)
├── docker/                  # Cấu hình Docker
├── models/
│   └── best_model.pkl       # Model tốt nhất sau khi train
├── notebooks/               # Jupyter notebooks EDA & thử nghiệm
├── src/
│   └── train.py             # Script huấn luyện và so sánh model
├── web/                     # Frontend web
├── requirements.txt
└── README.md
```

---

## Công nghệ sử dụng

| Nhóm | Thư viện |
|------|----------|
| Data | `pandas`, `numpy` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Hyperparameter Tuning | `optuna` |
| Model Serving | `fastapi`, `uvicorn` |
| Model Persistence | `joblib` |
| Containerization | `docker` |

---

## Cài đặt

**Yêu cầu:** Python 3.8+

```bash
# 1. Clone repo
git clone https://github.com/kietvoiuh/hcm-house-price-prediction.git
cd hcm-house-price-prediction

# 2. Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Cài dependencies
pip install -r requirements.txt
```

---

## Hướng dẫn sử dụng

### Bước 1 — Huấn luyện mô hình

```bash
python src/train.py
```

Script sẽ tự động:
- Load dữ liệu từ `data/processed/processed_house_data.csv`
- Train 3 mô hình: Linear Regression, Random Forest, XGBoost
- So sánh MAE, RMSE, R² Score
- Lưu model tốt nhất vào `models/best_model.pkl`

### Bước 2 — Chạy API

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API sẽ chạy tại `http://localhost:8000`

### Bước 3 — Kiểm tra API

Mở trình duyệt tại:
- **Swagger UI:** `http://localhost:8000/docs`
- **Health check:** `http://localhost:8000/health`

---

## API Reference

### `GET /health`

Kiểm tra trạng thái server.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /predict`

Dự đoán giá nhà dựa trên thông tin đầu vào.

**Request body:**

```json
{
  "Area": 60.0,
  "Bedrooms": 3,
  "Bathrooms": 2,
  "District": "Quận 7",
  "Latitude": 10.7354,
  "Longitude": 106.7218,
  "Legal": "Sổ hồng"
}
```

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `Area` | float | Diện tích (m²) |
| `Bedrooms` | int | Số phòng ngủ |
| `Bathrooms` | int | Số phòng tắm |
| `District` | string | Quận/Huyện tại TP.HCM |
| `Latitude` | float | Vĩ độ |
| `Longitude` | float | Kinh độ |
| `Legal` | string | Pháp lý: `Sổ đỏ`, `Sổ hồng`, `Hợp đồng`, `Giấy tờ khác` |

**Response:**

```json
{
  "predicted_price_million_vnd": 4250.50,
  "predicted_price_billion_vnd": 4.2505,
  "input_received": { ... }
}
```

**Ví dụ với curl:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Area": 60.0,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "District": "Quận 7",
    "Latitude": 10.7354,
    "Longitude": 106.7218,
    "Legal": "Sổ hồng"
  }'
```

---

## Mô hình & Kết quả

Ba mô hình được huấn luyện và so sánh tự động:

| Mô hình | MAE | RMSE | R² Score |
|---------|-----|------|----------|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| **XGBoost** | — | — | **—** |

> Kết quả cụ thể được in ra sau khi chạy `python src/train.py`. Model tốt nhất theo R² sẽ được lưu tự động.

**Cấu hình XGBoost:**
- `n_estimators`: 300
- `learning_rate`: 0.05
- `max_depth`: 8
- `subsample`: 0.8
- `colsample_bytree`: 0.8

**Cấu hình Random Forest:**
- `n_estimators`: 200
- `max_depth`: 15

---

## Docker

```bash
# Build image
docker build -f docker/Dockerfile -t hcm-house-price-api .

# Run container
docker run -p 8000:8000 hcm-house-price-api
```

---

## Tác giả

**kietvoiuh** — [GitHub](https://github.com/kietvoiuh/hcm-house-price-prediction)
