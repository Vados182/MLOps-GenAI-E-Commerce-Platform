import os
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="E-Commerce Churn Prediction API",
    description="REST API do predykcji ryzyka churnu klientów platformy Olist.",
    version="1.0.0"
)

MODEL_PATH = "best_xgboost_model.json"
model = xgb.XGBClassifier()

if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
    print(f"✓ Pomyślnie załadowano model z pliku: {MODEL_PATH}")
else:
    print(f"⚠️ Ostrzeżenie: Plik {MODEL_PATH} nie istnieje lokalnie!")


# Schemat dopasowany dokładnie do 9 cech modelu
class CustomerFeatures(BaseModel):
    actual_delivery_days: float = Field(..., description="Rzeczywisty czas dostawy w dniach", json_schema_extra={"example": 12.0})
    delivery_diff_days: float = Field(..., description="Różnica między szacowanym a faktycznym czasem dostawy", json_schema_extra={"example": -2.5})
    is_delayed: int = Field(..., description="Czy dostawa była opóźniona (1 lub 0)", json_schema_extra={"example": 0})
    items_count: int = Field(..., description="Liczba przedmiotów w zamówieniu", json_schema_extra={"example": 1})
    total_order_value: float = Field(..., description="Łączna wartość zamówienia", json_schema_extra={"example": 150.0})
    total_freight_value: float = Field(..., description="Łączny koszt dostawy", json_schema_extra={"example": 15.5})
    avg_product_weight: float = Field(..., description="Średnia waga produktu w gramach", json_schema_extra={"example": 500.0})
    avg_photos_qty: float = Field(..., description="Średnia liczba zdjęć produktu", json_schema_extra={"example": 2.0})
    avg_description_length: float = Field(..., description="Średnia długość opisu produktu", json_schema_extra={"example": 500.0})


class PredictionOutput(BaseModel):
    churn_probability: float
    is_high_risk: bool
    status: str


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Olist Churn Prediction API",
        "model_loaded": os.path.exists(MODEL_PATH)
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_churn(features: CustomerFeatures):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model nie został załadowany.")

    try:
        feature_order = [
            "actual_delivery_days",
            "delivery_diff_days",
            "is_delayed",
            "items_count",
            "total_order_value",
            "total_freight_value",
            "avg_product_weight",
            "avg_photos_qty",
            "avg_description_length"
        ]
        
        input_dict = features.model_dump()
        input_df = pd.DataFrame([input_dict])[feature_order]
        
        probability = float(model.predict_proba(input_df)[:, 1][0])
        is_high_risk = probability >= 0.5

        return PredictionOutput(
            churn_probability=round(probability, 4),
            is_high_risk=is_high_risk,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Błąd podczas predykcji: {str(e)}")