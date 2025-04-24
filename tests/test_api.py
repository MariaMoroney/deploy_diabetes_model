import os
import pytest
import json
import pandas as pd
import joblib
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from app.main import app
from app.model.train import load_data, train_model

# Cliente de prueba
client = TestClient(app)

# Directorio para datos de prueba
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Crear modelo de prueba si no existe ya
TEST_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_model.pkl")
TEST_DATA_PATH = os.path.join(TEST_DATA_DIR, "test_diabetes.csv")

# Verificar si necesitamos crear datos y modelo de prueba
if not os.path.exists(TEST_DATA_PATH):
    # Datos de prueba ampliados para evitar problemas con stratify
    TEST_DATA = pd.DataFrame({
        'Pregnancies': [1, 8, 1, 0, 5, 3, 10, 2, 4, 11],
        'Glucose': [89, 183, 89, 137, 116, 78, 115, 197, 110, 143],
        'BloodPressure': [66, 64, 66, 40, 74, 50, 0, 70, 92, 94],
        'SkinThickness': [23, 0, 94, 35, 0, 32, 0, 45, 0, 33],
        'Insulin': [94, 0, 0, 168, 0, 88, 0, 543, 0, 146],
        'BMI': [28.1, 23.3, 28.1, 43.1, 25.6, 31, 35.3, 30.5, 37.6, 36.6],
        'DiabetesPedigreeFunction': [0.167, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.191, 0.254],
        'Age': [21, 32, 21, 33, 30, 26, 29, 67, 30, 51],
        'Outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    # Guardar datos de prueba
    TEST_DATA.to_csv(TEST_DATA_PATH, index=False)

if not os.path.exists(TEST_MODEL_PATH):
    # Cargar datos
    X, y = load_data(TEST_DATA_PATH)

    # Preprocesar datos (sin usar la función original que tiene el problema)
    # Manejar valores faltantes o cero
    features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Reemplazar ceros con NaN
    for feature in features_with_zeros:
        X.loc[X[feature] == 0, feature] = float('nan')

    # Imputar los valores NaN
    for feature in features_with_zeros:
        median_value = X[feature].median()
        X[feature] = X[feature].fillna(median_value)

    # División simple sin estratificación para tests
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=None
    )

    # Entrenar modelo
    model = train_model(X_train, y_train)

    # Guardar modelo
    joblib.dump(model, TEST_MODEL_PATH)

# Sobrescribir la dependencia de get_classifier en los endpoints
import app.api.endpoints as endpoints
from app.model.diabetes_classifier import DiabetesClassifier


# Modificar la función de dependencia para usar el modelo de prueba
def get_test_classifier():
    return DiabetesClassifier(model_path=TEST_MODEL_PATH)


# Reemplazar la dependencia original con la de prueba
endpoints.get_classifier = get_test_classifier


def test_root():
    """Probar el endpoint raíz."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    """Probar el endpoint de health check."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_root():
    """Probar el endpoint raíz de la API."""
    response = client.get("/api/v1/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_endpoint_valid():
    """Probar el endpoint de predicción con datos válidos."""
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert "probability" in result
    assert "diagnosis" in result

    # Verificar tipos de datos
    assert isinstance(result["prediction"], int)
    assert isinstance(result["probability"], float)
    assert isinstance(result["diagnosis"], str)

    # Verificar valores válidos
    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1
    assert result["diagnosis"] in ["Diabetes", "No Diabetes"]


def test_predict_endpoint_invalid():
    """Probar el endpoint de predicción con datos inválidos."""
    # Datos con una característica faltante
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        # Falta BloodPressure
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422  # Error de validación


def test_batch_predict_endpoint_valid():
    """Probar el endpoint de predicción por lotes con datos válidos."""
    payload = {
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
    response = client.post("/api/v1/batch-predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 2

    # Verificar estructura de cada predicción
    for prediction in result["predictions"]:
        assert "prediction" in prediction
        assert "probability" in prediction
        assert "diagnosis" in prediction

        # Verificar tipos de datos
        assert isinstance(prediction["prediction"], int)
        assert isinstance(prediction["probability"], float)
        assert isinstance(prediction["diagnosis"], str)

        # Verificar valores válidos
        assert prediction["prediction"] in [0, 1]
        assert 0 <= prediction["probability"] <= 1
        assert prediction["diagnosis"] in ["Diabetes", "No Diabetes"]


def test_batch_predict_endpoint_invalid():
    """Probar el endpoint de predicción por lotes con datos inválidos."""
    # Datos con una entrada válida y otra inválida
    payload = {
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
                # Falta BloodPressure
                "SkinThickness": 29,
                "Insulin": 0,
                "BMI": 26.6,
                "DiabetesPedigreeFunction": 0.351,
                "Age": 31
            }
        ]
    }
    response = client.post("/api/v1/batch-predict", json=payload)
    assert response.status_code == 422  # Error de validación