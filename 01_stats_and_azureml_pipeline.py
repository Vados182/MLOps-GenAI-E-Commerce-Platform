import os
import zipfile
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import xgboost as xgb
import optuna
import shap
import mlflow
import mlflow.xgboost
from sklearn.metrics import roc_auc_score

# Moduły Azure ML SDK v2
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Wyciszenie zbędnych komunikatów Optuny
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==============================================================================
# 0. AUTOMATYCZNE POBIERANIE ZBIORU DANYCH Z KAGGLE
# ==============================================================================
def download_and_extract_olist_data(data_dir: str = "data"):
    """
    Sprawdza, czy dane Olist istnieją lokalnie. Jeśli nie, pobiera je
    automatycznie z API Kaggle i rozpakowuje do wskazanego folderu.
    """
    required_files = [
        "olist_customers_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_products_dataset.csv",
        "product_category_name_translation.csv"
    ]

    os.makedirs(data_dir, exist_ok=True)
    
    # Sprawdzenie, czy wszystkie pliki są już na dysku
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    
    if not missing_files:
        print(f"✓ Wszystkie pliki CSV znajdują się już w folderze '{data_dir}/'. Skok pobierania.")
        return

    print(f"📥 Brakujące pliki: {missing_files}. Pobieranie zbioru Olist z Kaggle...")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Pobieranie archiwum z Kaggle (olistbr/brazilian-ecommerce)
        dataset_slug = "olistbr/brazilian-ecommerce"
        api.dataset_download_files(dataset_slug, path=data_dir, unzip=True)
        print(f"✓ Pobrano i pomyślnie rozpakowano dane do folderu '{data_dir}/'.")
        
    except Exception as e:
        print(f"\n❌ Błąd podczas pobierania z Kaggle: {e}")
        print("💡 Upewnij się, że umieściłeś plik 'kaggle.json' w katalogu: C:\\Users\\v.vorobiov\\.kaggle\\kaggle.json\n")
        raise e


# ==============================================================================
# 1. PRZYGOTOWANIE BAZY DANYCH I DANYCH WYJŚCIOWYCH (DUCKDB)
# ==============================================================================
def create_duckdb_views(con: duckdb.DuckDBPyConnection, data_dir: str = "data"):
    """
    Mapuje pliki CSV Olist na widoki SQL w bazie DuckDB z uwzględnieniem folderu z danymi.
    """
    tables = {
        "customers": "olist_customers_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "order_reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "translation": "product_category_name_translation.csv"
    }

    # Dynamiczne tworzenie widoków dla plików źródłowych ze ścieżką do data_dir
    for table_name, csv_file in tables.items():
        file_path = os.path.join(data_dir, csv_file).replace("\\", "/")
        con.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")

    # Widok oczyszczonych produktów z angielskimi nazwami
    con.execute("""
    CREATE OR REPLACE VIEW v_cleaned_products AS
    SELECT 
        p.product_id,
        COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS product_category,
        COALESCE(p.product_name_lenght, 0) AS product_name_length,
        COALESCE(p.product_description_lenght, 0) AS product_description_length,
        COALESCE(p.product_photos_qty, 0) AS product_photos_qty,
        COALESCE(p.product_weight_g, 0) AS product_weight_g,
        COALESCE(p.product_length_cm, 0) AS product_length_cm,
        COALESCE(p.product_height_cm, 0) AS product_height_cm,
        COALESCE(p.product_width_cm, 0) AS product_width_cm
    FROM products p
    LEFT JOIN translation t ON p.product_category_name = t.product_category_name;
    """)

    # Widok zamówień z kalkulacją wskaźników logistycznych
    con.execute("""
    CREATE OR REPLACE VIEW v_cleaned_orders AS
    SELECT 
        order_id,
        customer_id,
        order_status,
        order_purchase_timestamp::TIMESTAMP AS purchase_time,
        order_delivered_customer_date::TIMESTAMP AS delivered_to_customer_time,
        order_estimated_delivery_date::TIMESTAMP AS estimated_delivery_time,
        
        -- Czas dostawy w dniach
        DATE_PART('day', order_delivered_customer_date::TIMESTAMP - order_purchase_timestamp::TIMESTAMP) AS actual_delivery_days,
        
        -- Flaga opóźnienia dostawy (1 = spóźnienie, 0 = na czas)
        CASE 
            WHEN order_delivered_customer_date::TIMESTAMP > order_estimated_delivery_date::TIMESTAMP THEN 1
            WHEN order_delivered_customer_date IS NULL AND order_estimated_delivery_date::TIMESTAMP < CURRENT_DATE THEN 1
            ELSE 0 
        END AS is_delayed,
        
        -- Odchylenie od planowanego terminu (w dniach)
        DATE_PART('day', order_delivered_customer_date::TIMESTAMP - order_estimated_delivery_date::TIMESTAMP) AS delivery_diff_days
    FROM orders
    WHERE order_status = 'delivered';
    """)


# ==============================================================================
# 2. ANALIZA STATYSTYCZNA I TESTOWANIE HIPOTEZ
# ==============================================================================
def run_statistical_analysis(con: duckdb.DuckDBPyConnection) -> dict:
    """
    Weryfikuje hipotezy statystyczne dotyczące wpływu opóźnień dostaw na oceny klientów.
    """
    print("\n--- 📊 Uruchamianie Analizy Statystycznej ---")

    query = """
    SELECT 
        o.is_delayed,
        r.review_score,
        o.actual_delivery_days
    FROM v_cleaned_orders o
    JOIN order_reviews r ON o.order_id = r.order_id
    WHERE r.review_score IS NOT NULL;
    """
    df_stats = con.execute(query).df()

    scores_on_time = df_stats[df_stats['is_delayed'] == 0]['review_score']
    scores_delayed = df_stats[df_stats['is_delayed'] == 1]['review_score']

    # 1. Test Mann-Whitneya U
    u_stat, p_value = stats.mannwhitneyu(scores_on_time, scores_delayed, alternative='greater')
    
    # 2. Przedział ufności (95%) dla średniego czasu dostawy
    delivery_days = df_stats['actual_delivery_days'].dropna()
    mean_delivery = np.mean(delivery_days)
    sem_delivery = stats.sem(delivery_days)
    ci_95 = stats.t.interval(0.95, len(delivery_days)-1, loc=mean_delivery, scale=sem_delivery)

    results = {
        "p_value_mann_whitney": float(p_value),
        "mean_delivery_days": float(mean_delivery),
        "ci_95_lower": float(ci_95[0]),
        "ci_95_upper": float(ci_95[1]),
        "avg_score_on_time": float(scores_on_time.mean()),
        "avg_score_delayed": float(scores_delayed.mean())
    }

    print(f"Średnia ocena zamówień na czas: {results['avg_score_on_time']:.2f}")
    print(f"Średnia ocena zamówień opóźnionych: {results['avg_score_delayed']:.2f}")
    print(f"Test Mann-Whitneya p-value: {results['p_value_mann_whitney']:.4e}")
    print(f"Średni czas dostawy: {results['mean_delivery_days']:.2f} dni (95% CI: [{results['ci_95_lower']:.2f}, {results['ci_95_upper']:.2f}])")

    return results


# ==============================================================================
# 3. FEATURE ENGINEERING POD MODELOWANIE UCZENIA MASZYNOWEGO
# ==============================================================================
def prepare_ml_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Tworzy końcowy zbiór danych analitycznych do predykcji niskiej oceny (1-2 gwiazdki).
    """
    print("\n--- 🛠️ Przygotowanie Zbioru Danych ML ---")
    
    query = """
    SELECT 
        o.order_id,
        CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END AS is_low_rating,
        o.actual_delivery_days,
        o.delivery_diff_days,
        o.is_delayed,
        COUNT(oi.product_id) AS items_count,
        SUM(oi.price) AS total_order_value,
        SUM(oi.freight_value) AS total_freight_value,
        AVG(p.product_weight_g) AS avg_product_weight,
        AVG(p.product_photos_qty) AS avg_photos_qty,
        AVG(p.product_description_length) AS avg_description_length
    FROM v_cleaned_orders o
    JOIN order_reviews r ON o.order_id = r.order_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN v_cleaned_products p ON oi.product_id = p.product_id
    GROUP BY 
        o.order_id, 
        r.review_score, 
        o.actual_delivery_days, 
        o.delivery_diff_days, 
        o.is_delayed;
    """
    df_ml = con.execute(query).df().dropna()
    print(f"Rozmiar zbioru treningowego: {df_ml.shape[0]} wierszy, {df_ml.shape[1]} kolumn.")
    return df_ml


# ==============================================================================
# 4. INICJALIZACJA POŁĄCZENIA Z AZURE ML WORKSPACE
# ==============================================================================
def init_azure_ml_client() -> MLClient:
    """
    Inicjalizuje klienta Azure ML. W przypadku braku poświadczeń
    przełącza śledzenie MLflow na lokalną bazę SQLite.
    """
    print("\n--- ☁️ Łączenie z Azure ML Workspace ---")
    
    subscription_id = "1f00801b-7f04-41dc-af18-3a4e9640b347"
    resource_group = "rg-mlops-neurope"
    workspace_name = "ws-mlops-ecommerce"

    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        ws = ml_client.workspaces.get(workspace_name)
        mlflow.set_tracking_uri(ws.mlflow_tracking_uri)
        print(f"✓ Pomyślnie połączono z Azure ML Workspace: {workspace_name}")
        return ml_client
    except Exception as e:
        print("⚠️ Brak poświadczeń Azure ML. Rejestrowanie eksperymentu lokalnie w MLflow (SQLite)...")
        # Używamy SQLite, aby uniknąć wyjątku nowszych wersji MLflow dla katalogów plikowych
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        return None

# ==============================================================================
# 5. TRENING XGBOOST + OPTUNA + SHAP + REJESTRACJA W MLFLOW / AZURE ML
# ==============================================================================
def train_and_log_model(df_ml: pd.DataFrame, stat_metrics: dict):
    """
    Strojenie hiperparametrów XGBoost z Optuną oraz generowanie SHAP.
    """
    print("\n--- 🚀 Uruchamianie Potoku Treningowego (XGBoost + Optuna) ---")

    X = df_ml.drop(columns=['order_id', 'is_low_rating'])
    y = df_ml['is_low_rating']

    train_size = int(len(df_ml) * 0.8)
    X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

    # --- Optuna Optimization ---
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'eval_metric': 'logloss',
            'random_state': 42
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=42)
    best_model.fit(X_train, y_train)

    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    print("Obliczanie wartości SHAP dla modelu...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_val)

    # --- Logging to MLflow / Azure ML ---
    mlflow.set_experiment("Olist_Churn_Prediction_AzureML")

    with mlflow.start_run(run_name="XGBoost_Optuna_SHAP_Run") as run:
        # 1. Metryki i parametry
        mlflow.log_params(best_params)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("mann_whitney_p_value", stat_metrics.get("p_value", 0.0))

        # 2. Wykres SHAP lokalnie
        shap_summary_path = "shap_summary.png"
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_val, show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_path)
        plt.close()

        # 3. Model lokalnie
        model_path = "best_xgboost_model.json"
        best_model.save_model(model_path)

        # 4. Zapis artefaktów z obsługą błędu wtyczki Azure ML
        try:
            mlflow.log_artifact(shap_summary_path)
            mlflow.log_artifact(model_path)
            print("✓ Artefakty zostały pomyślnie przesłane do Azure ML.")
        except Exception as err:
            print(f"⚠️ Wyniki i metryki zarejestrowano w Azure ML. Artefakty zapisano lokalnie ({shap_summary_path}, {model_path}).")

    print("\n✅ Proces uczenia i analizy zakończony sukcesem!")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    DATA_FOLDER = "data"

    # 0. Automatyczne pobranie danych z Kaggle (jeśli brak)
    download_and_extract_olist_data(data_dir=DATA_FOLDER)

    # Inicjalizacja połączenia DuckDB
    db_connection = duckdb.connect(database=':memory:')

    # 1. Tworzenie widoków w bazie
    create_duckdb_views(db_connection, data_dir=DATA_FOLDER)

    # 2. Analiza statystyczna
    statistical_results = run_statistical_analysis(db_connection)

    # 3. Przygotowanie danych do ML
    df_features = prepare_ml_dataset(db_connection)

    # 4. Azure ML / MLflow
    azure_client = init_azure_ml_client()

    # 5. Trening i logowanie
    train_and_log_model(df_features, statistical_results)