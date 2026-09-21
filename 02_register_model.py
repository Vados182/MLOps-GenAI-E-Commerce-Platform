import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# ------------------------------------------------------------------------------
# 1. KONFIGURACJA I POŁĄCZENIE Z AZURE ML
# ------------------------------------------------------------------------------
SUBSCRIPTION_ID = "1f00801b-7f04-41dc-af18-3a4e9640b347"
RESOURCE_GROUP = "rg-mlops-neurope"
WORKSPACE_NAME = "ws-mlops-ecommerce"
MODEL_LOCAL_PATH = "best_xgboost_model.json"
MODEL_NAME = "olist_churn_xgboost"


def register_model():
    print("--- ☁️ Łączenie z Azure ML Workspace ---")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    print(f"✓ Połączono z workspace: {ml_client.workspace_name}")

    # Sprawdzenie czy plik modelu istnieje lokalnie
    if not os.path.exists(MODEL_LOCAL_PATH):
        raise FileNotFoundError(
            f"❌ Nie znaleziono pliku {MODEL_LOCAL_PATH}! "
            "Uruchom najpierw skrypt 01_stats_and_azureml_pipeline.py, aby wygenerować model."
        )

    # --------------------------------------------------------------------------
    # 2. DEFINICJA I REJESTRACJA MODELU W REJESTRZE CHMUROWYM
    # --------------------------------------------------------------------------
    print(f"\n--- 📦 Rejestracja modelu '{MODEL_NAME}' w Azure ML Model Registry ---")
    
    model_asset = Model(
        path=MODEL_LOCAL_PATH,
        name=MODEL_NAME,
        description="Model XGBoost zoptymalizowany Optuną do predykcji churnu klientów Olist.",
        type=AssetTypes.CUSTOM_MODEL
    )

    registered_model = ml_client.models.create_or_update(model_asset)

    print("\n✅ Model został pomyślnie zarejestrowany!")
    print(f"   • Nazwa w Azure: {registered_model.name}")
    print(f"   • Wersja: {registered_model.version}")
    print(f"   • Ścieżka zasobu: {registered_model.id}")


if __name__ == "__main__":
    register_model()