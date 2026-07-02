import os
import time
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.tracking import MlflowClient

# Настройка MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")

# --- Ждём, пока MLflow станет доступен (retry) ---
max_retries = 30
for i in range(max_retries):
    try:
        client = MlflowClient()
        client.search_experiments()  # лёгкий ping
        print("✅ MLflow доступен")
        break
    except Exception as e:
        print(f"⏳ Ожидание MLflow... ({i+1}/{max_retries})")
        time.sleep(2)
else:
    raise RuntimeError("❌ MLflow не доступен после 60 секунд ожидания")

# --- Проверяем, есть ли уже модель в Production ---
try:
    versions = client.get_latest_versions("DiabetesLinearRegression", stages=["Production"])
    if versions:
        print("✅ Модель уже в Production, пропускаем обучение")
        exit(0)
except Exception:
    pass  # Модели ещё нет, обучаем

# Загрузка данных
data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Логирование в MLflow
mlflow.set_experiment("diabetes_prediction")
model_name = "DiabetesLinearRegression"

with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="diabetes_model",
        registered_model_name=model_name,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
    )

# Перевод в Production
latest_versions = client.get_latest_versions(model_name, stages=None)
if latest_versions:
    latest_version = max(int(v.version) for v in latest_versions)
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Модель версии {latest_version} переведена в Production")
else:
    print("❌ Модель не найдена в реестре")

print("✅ Готово")