import os
import numpy as np
import mlflow
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput, RequestInput
from mlserver.codecs import NumpyCodec

# Настройка MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"))
os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"))


class DiabetesRuntime(MLModel):
    async def load(self) -> bool:
        """Загрузка модели из MLflow Registry при старте MLServer"""
        model_name = "DiabetesLinearRegression"
        model_stage = "Production"

        self._model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_stage}")
        print(f"✅ Модель {model_name}/{model_stage} загружена в MLServer")
        return True

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """Предсказание с правильной обработкой входных данных"""

        # Берём первый input
        request_input = payload.inputs[0]

        # Декодируем данные через NumpyCodec (поддерживает разные форматы)
        data = NumpyCodec.decode_input(request_input)

        # Убеждаемся, что данные 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Предсказание
        prediction = self._model.predict(data)

        # Формируем ответ
        return InferenceResponse(
            model_name=self.name,
            outputs=[
                ResponseOutput(
                    name="predict",
                    shape=list(prediction.shape),
                    datatype="FP32",
                    data=prediction.tolist()
                )
            ]
        )