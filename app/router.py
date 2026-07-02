import os
import logging
import httpx
from fastapi import APIRouter, HTTPException
from app.schemas import Predict, PatientFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

MLSERVER_URL = os.getenv("MLSERVER_URL", "http://mlserver:8080")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-model")


@router.post("/predict", response_model=Predict)
async def predict(patient: PatientFeatures):
    logger.info(f"Получен запрос на предсказание: {patient.features}")

    # V2 Inference Protocol — данные как flat list, shape указывает размерность
    payload = {
        "inputs": [
            {
                "name": "predict",
                "shape": [1, len(patient.features)],
                "datatype": "FP32",
                "data": patient.features  # flat list, не двойной!
            }
        ]
    }

    url = f"{MLSERVER_URL}/v2/models/{MODEL_NAME}/infer"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            outputs = result.get("outputs", [])
            if not outputs:
                raise HTTPException(status_code=500, detail="No outputs from MLServer")

            prediction_data = outputs[0].get("data", [])
            if not prediction_data:
                raise HTTPException(status_code=500, detail="Empty prediction data")

            prediction_value = float(prediction_data[0])
            logger.info(f"Предсказание: {prediction_value}")
            return Predict(prediction=prediction_value)

    except httpx.TimeoutException:
        logger.error("Timeout connecting to MLServer")
        raise HTTPException(status_code=504, detail="MLServer timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"MLServer error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLServer error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "OK"}