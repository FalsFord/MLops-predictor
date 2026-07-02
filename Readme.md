# 🚀 Diabetes Prediction API

Сервис для предсказания прогрессии диабета на основе 10 клинических признаков.  
Архитектура: **FastAPI** → **MLServer** (V2 Inference Protocol) → **MLflow** (Model Registry) + **MinIO** (артефакты) +\*
*PostgreSQL*\* (метаданные).

---

## 📋 Содержание

* [Архитектура](#архитектура)
* [Технологический стек](#технологический-стек)
* [Структура проекта](#структура-проекта)
* [Как запустить](№как-запустить)
* [Эндпоинты](#эндпоинты)
* [Примеры запросов](#примеры-запросов)
* [MLflow UI](#mlflow-ui)
* [Разработка](#разработка)
* 

---

## 🏗 Архитектура

```
┌─────────────┐      V2 Protocol       ┌─────────────┐      ┌─────────────┐
│   Client    │ ─────────────────────→ │  FastAPI    │ ───→ │   MLServer  │
└─────────────┘                        └─────────────┘      └──────┬──────┘
                                                                    │
                                                                    ↓
                                                             ┌─────────────┐
                                                             │   MLflow    │
                                                             │   Registry  │
                                                             └──────┬──────┘
                                                                    │
                    ┌─────────────┐      ┌─────────────┐         │
                    │  PostgreSQL │ ←─── │  MLflow     │ ←─────┘
                    │  (metadata) │      │  Server     │
                    └─────────────┘      └──────┬──────┘
                                                │
                                         ┌─────────────┐
                                         │    MinIO    │
                                         │ (artifacts) │
                                         └─────────────┘
```

**Поток данных:**

1. `trainer` обучает модель `LinearRegression` и регистрирует её в MLflow → stage **Production**
2. `mlserver` при старте загружает модель из MLflow Registry через кастомный runtime
3. `fastapi-app` принимает запросы, проксирует их в MLServer по V2 Inference Protocol
4. `mlflow-server` + `postgres` + `minio` хранят эксперименты, метрики и артефакты

---

## 🛠 Технологический стек

|Компонент|Технология|Назначение|
|-|-|-|
|**API Gateway**|FastAPI + Uvicorn|REST API, валидация, проксирование|
|**Inference Server**|MLServer 1.7.0 + кастомный runtime|V2 Inference Protocol, предсказания|
|**ML Platform**|MLflow|Реестр моделей, эксперименты, метрики|
|**ML Backend**|scikit-learn 1.7.2|Обучение линейной регрессии|
|**Object Storage**|MinIO|Хранение артефактов моделей (S3-совместимый)|
|**Database**|PostgreSQL 15|Хранение метаданных MLflow|
|**Orchestration**|Docker Compose|Запуск всего стека|

---

## 📁 Структура проекта

```
.
├── app/
│   ├── __init__.py
│   ├── router.py              # Эндпоинты /predict и /health
│   └── schemas.py             # Pydantic-схемы запросов/ответов
├── mlserver_config/
│   ├── model-settings.json    # Конфиг MLServer (кастомный runtime)
│   └── models.py              # DiabetesRuntime — загрузка модели из MLflow
├── main.py                    # Точка входа FastAPI
├── train_model.py             # Обучение модели и публикация в MLflow
├── docker-compose.yml         # Полный стек инфраструктуры
├── Dockerfile                 # FastAPI-сервис
├── Dockerfile.mlserver        # MLServer с кастомным runtime
├── Dockerfile.trainer         # Контейнер для обучения модели
├── requirements.txt           # Зависимости Python
├── .gitignore
├── .dockerignore
└── README.md
```

---

## ⚡Как запустить

### 1\. Клонирование

```bash
git clone https://github.com/FalsFord/MLops-predictor.git
cd diabetes-predictor
```

### 2\. Запуск всего стека

```bash
docker compose up --build
```

**Что произойдёт:**

* Поднимутся `postgres`, `minio`, `mlflow`
* `minio-setup` создаст S3-бакет `mlflow-artifacts`
* `trainer` обучит модель и переведёт её в `Production`
* `mlserver` загрузит модель из MLflow Registry
* `fastapi-app` станет доступен на `http://localhost:8000`

> **Примечание:** при первом запуске `trainer` дождётся доступности MLflow (retry-логика внутри скрипта). При
> повторных запусках, если модель уже в `Production`, обучение пропускается.

### 3\. Проверка работоспособности

```bash
curl http://localhost:8000/health
```

**Ожидаемый ответ:**

```json
{
  "status": "OK"
}
```

### 4\. Предсказание

```bash
curl -X POST http://localhost:8000/predict \\\\
  -H "Content-Type: application/json" \\\\
  -d '{"features": \\\[0.1, -0.24, 0.55, 0.01, -0.04, 0.02, 0.06, 0.08, 0.02, 0.01]}'
```

**Ожидаемый ответ:**

```json
{
  "prediction": 609.6824265173714
}
```

---

## 📡 Эндпоинты

|Эндпоинт|Метод|Описание|
|-|-|-|
|`/health`|GET|Проверка работоспособности сервиса|
|`/predict`|POST|Предсказание прогрессии диабета по 10 признакам|
|`/docs`|GET|Swagger UI (интерактивная документация)|
|`/redoc`|GET|ReDoc (альтернативная документация)|

### Параметры `POST /predict`

|Поле|Тип|Описание|
|-|-|-|
|`features`|`list[float]`|Массив из **10 чисел** (нормализованные признаки датасета Diabetes)|

---

## 🔍 MLflow UI

Интерфейс MLflow доступен по адресу:  
**http://localhost:5000**

Там можно посмотреть:

* Эксперимент `diabetes_prediction`
* Зарегистрированную модель `DiabetesLinearRegression`
* Метрики (MSE, R²) и артефакты

---

## 🧪 Примеры запросов

### Проверка здоровья

```bash
curl http://localhost:8000/health
```

### Предсказание

```bash
curl -X POST http://localhost:8000/predict \\\\
  -H "Content-Type: application/json" \\\\
  -d '{"features": \\\[0.1, -0.24, 0.55, 0.01, -0.04, 0.02, 0.06, 0.08, 0.02, 0.01]}'
```

### Через Swagger UI

Откройте `http://localhost:8000/docs` → разверните `/predict` → **Try it out**

---

## 🛠 Разработка

### Локальный запуск (без Docker)

Требуется запущенный MLflow сервер и доступ к S3/MinIO.

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение модели (один раз)
python train\\\_model.py

# Запуск FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Пересборка после изменений

```bash
docker compose down
docker compose up --build
```

### Полная очистка (удаление volumes)

```bash
docker compose down -v
```

---

