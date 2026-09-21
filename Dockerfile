# 1. Lekki obraz bazowy Pythona
FROM python:3.11-slim

# 2. Ustawienie katalogu roboczego
WORKDIR /app

# 3. Instalacja bibliotek systemowych wymaganych m.in. przez XGBoost (libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt-get/lists/*

# 4. Kopiowanie i instalacja zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Kopiowanie kodu aplikacji FastAPI oraz zapisanego modelu
COPY 03_app_fastapi.py .
COPY best_xgboost_model.json .

# 6. Udostępnienie portu 8000
EXPOSE 8000

# 7. Uruchomienie aplikacji za pomocą serwera Uvicorn
CMD ["uvicorn", "03_app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]