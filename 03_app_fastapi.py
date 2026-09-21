import os
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(
    title="MLOps & GenAI E-Commerce Platform",
    description="REST API do predykcji ryzyka churnu oraz generowania spersonalizowanych akcji retencyjnych GenAI.",
    version="1.1.0"
)

MODEL_PATH = "best_xgboost_model.json"
model = xgb.XGBClassifier()

if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
    print(f"✓ Pomyślnie załadowano model z pliku: {MODEL_PATH}")

# Inicjalizacja klienta LLM (pobiera OPENAI_API_KEY lub GROQ_API_KEY ze środowiska)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


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


class RetentionOutput(BaseModel):
    churn_probability: float
    is_high_risk: bool
    retention_strategy: str
    status: str


def generate_genai_retention_offer(features: CustomerFeatures, probability: float) -> str:
    """Generuje spersonalizowaną ofertę retencyjną używając LLM."""
    if not client:
        return "Brak skonfigurowanego klucza API dla modelu GenAI. Skonfiguruj OPENAI_API_KEY w środowisku."

    prompt = f"""
    Jesteś ekspertem ds. retencji klientów w sklepie e-commerce Olist.
    Model ML wykrył wysokie ryzyko odejścia klienta (Prawdopodobieństwo churnu: {probability:.2%}).

    Kontekst klienta:
    - Ostatni czas dostawy: {features.actual_delivery_days} dni (opóźnienie: {features.is_delayed})
    - Różnica względem szacowanego czasu: {features.delivery_diff_days} dni
    - Liczba produktów w zamówieniu: {features.items_count}
    - Wartość zamówienia: {features.total_order_value} BRL (koszt dostawy: {features.total_freight_value} BRL)

    Napisz krótką (max 2-3 zdania), spersonalizowaną i empatyczną wiadomość do klienta z propozycją dedykowanej rekompensaty lub zniżki, aby zachęcić go do ponownych zakupów.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Nie udało się wygenerować oferty GenAI: {str(e)}"


@app.post("/predict-with-retention", response_model=RetentionOutput)
def predict_churn_and_retain(features: CustomerFeatures):
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

        retention_offer = "Klient w grupie niskiego ryzyka – brak konieczności akcji retencyjnej."
        if is_high_risk:
            retention_offer = generate_genai_retention_offer(features, probability)

        return RetentionOutput(
            churn_probability=round(probability, 4),
            is_high_risk=is_high_risk,
            retention_strategy=retention_offer,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Błąd podczas predykcji: {str(e)}")