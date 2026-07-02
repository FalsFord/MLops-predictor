from pydantic import Field, BaseModel
from typing import List


class Predict(BaseModel):
    prediction: float = Field(description="Предсказание")


class PatientFeatures(BaseModel):
    features: List[float] = Field(
        ...,
        description="Список признаков для предсказания",
        example=[0.1, -0.24, 0.55, 0.01, -0.04, 0.02, 0.06, 0.08, 0.02, 0.01],
    )
