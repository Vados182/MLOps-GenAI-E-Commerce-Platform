# 🛒 MLOps-GenAI-E-Commerce-Platform

![CI/CD Pipeline](https://github.com/Vados182/MLOps-GenAI-E-Commerce-Platform/actions/workflows/cicd.yml/badge.svg)

End-to-end MLOps pipeline to predict customer churn risk for the Olist E-Commerce platform using XGBoost, Azure ML, FastAPI, Docker, and Render.

## 🚀 Live Demo & API Documentation
- **Swagger UI**: [https://olist-churn-api.onrender.com/docs](https://olist-churn-api.onrender.com/docs)
- **Health Check**: `GET https://olist-churn-api.onrender.com/`

---

## 🛠️ Tech Stack & Architecture
- **ML Framework**: XGBoost, Scikit-Learn
- **Hyperparameter Tuning**: Optuna
- **Experiment Tracking & Registry**: Azure Machine Learning Workspace
- **REST API**: FastAPI, Pydantic v2, Uvicorn
- **Containerization & Deployment**: Docker, Render Cloud Platform

---

## 📊 Features Used for Prediction (9 Key Variables)
1. `actual_delivery_days` – Actual delivery time in days
2. `delivery_diff_days` – Difference between estimated and actual delivery date
3. `is_delayed` – Binary flag indicating delivery delay (0 or 1)
4. `items_count` – Total number of items in order
5. `total_order_value` – Total value of the order
6. `total_freight_value` – Freight (shipping) cost
7. `avg_product_weight` – Average product weight in grams
8. `avg_photos_qty` – Average product photos count
9. `avg_description_length` – Average description character length

---

## 🔌 API Example Request

```json
POST /predict
{
  "actual_delivery_days": 12.0,
  "delivery_diff_days": -2.5,
  "is_delayed": 0,
  "items_count": 1,
  "total_order_value": 150.0,
  "total_freight_value": 15.5,
  "avg_product_weight": 500.0,
  "avg_photos_qty": 2.0,
  "avg_description_length": 500.0
}