from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import numpy as np

from app.model.diabetes_classifier import DiabetesClassifier

# Crear el router
router = APIRouter()


# Modelo de datos para la entrada
class DiabetesFeatures(BaseModel):
    """
    Modelo de datos para las características de entrada para la predicción de diabetes.
    """
    Pregnancies: int = Field(..., ge=0, description="Número de embarazos")
    Glucose: float = Field(..., ge=0, description="Concentración de glucosa en plasma (mg/dL)")
    BloodPressure: float = Field(..., ge=0, description="Presión arterial diastólica (mm Hg)")
    SkinThickness: float = Field(..., ge=0, description="Grosor del pliegue cutáneo del tríceps (mm)")
    Insulin: float = Field(..., ge=0, description="Insulina sérica de 2 horas (mu U/ml)")
    BMI: float = Field(..., ge=0, description="Índice de masa corporal (peso en kg/(altura en m)^2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Función de pedigrí de diabetes")
    Age: int = Field(..., ge=21, description="Edad (años)")

    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }
        }


# Modelo de datos para múltiples entradas
class BatchDiabetesFeatures(BaseModel):
    """
    Modelo de datos para múltiples conjuntos de características para predicciones por lotes.
    """
    inputs: List[DiabetesFeatures]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Pregnancies": 6,
                        "Glucose": 148,
                        "BloodPressure": 72,
                        "SkinThickness": 35,
                        "Insulin": 0,
                        "BMI": 33.6,
                        "DiabetesPedigreeFunction": 0.627,
                        "Age": 50
                    },
                    {
                        "Pregnancies": 1,
                        "Glucose": 85,
                        "BloodPressure": 66,
                        "SkinThickness": 29,
                        "Insulin": 0,
                        "BMI": 26.6,
                        "DiabetesPedigreeFunction": 0.351,
                        "Age": 31
                    }
                ]
            }
        }


# Modelo de datos para la respuesta
class PredictionResponse(BaseModel):
    """
    Modelo de datos para la respuesta de la predicción.
    """
    prediction: int
    probability: float
    diagnosis: str


# Modelo de datos para respuesta por lotes
class BatchPredictionResponse(BaseModel):
    """
    Modelo de datos para la respuesta de predicciones por lotes.
    """
    predictions: List[Dict[str, Any]]


# Dependencia para obtener una instancia del clasificador
def get_classifier():
    """
    Dependencia para obtener una instancia del clasificador de diabetes.

    Returns:
        DiabetesClassifier: Instancia del clasificador.
    """
    return DiabetesClassifier()


@router.get("/", response_model=Dict[str, str])
async def root():
    """
    Endpoint raíz para verificar que la API está funcionando.

    Returns:
        dict: Mensaje de estado.
    """
    return {"message": "API de Predicción de Diabetes"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(
        features: DiabetesFeatures,
        classifier: DiabetesClassifier = Depends(get_classifier)
):
    """
    Endpoint para realizar una predicción individual.

    Args:
        features (DiabetesFeatures): Características del paciente.
        classifier (DiabetesClassifier): Instancia del clasificador.

    Returns:
        PredictionResponse: Respuesta con la predicción.
    """
    try:
        # Convertir el modelo Pydantic a un diccionario
        features_dict = features.dict()

        # Realizar la predicción
        result = classifier.predict(features_dict)

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(
        batch_features: BatchDiabetesFeatures,
        classifier: DiabetesClassifier = Depends(get_classifier)
):
    """
    Endpoint para realizar predicciones por lotes.

    Args:
        batch_features (BatchDiabetesFeatures): Lista de conjuntos de características.
        classifier (DiabetesClassifier): Instancia del clasificador.

    Returns:
        BatchPredictionResponse: Respuesta con las predicciones por lotes.
    """
    try:
        # Convertir los modelos Pydantic a una lista de diccionarios
        features_list = [features.dict() for features in batch_features.inputs]

        # Realizar las predicciones por lotes
        results = classifier.batch_predict(features_list)

        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Endpoint para verificar el estado de salud de la API.

    Returns:
        dict: Estado de salud de la API.
    """
    return {"status": "healthy"}