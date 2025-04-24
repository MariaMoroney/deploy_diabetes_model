import os
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from app.model.diabetes_classifier import DiabetesClassifier
from app.model.train import load_data, train_model, evaluate_model

# Ruta al directorio de datos de prueba
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Datos de prueba ampliados
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

# Guardar datos de prueba en un archivo CSV
TEST_DATA_PATH = os.path.join(TEST_DATA_DIR, "test_diabetes.csv")
TEST_DATA.to_csv(TEST_DATA_PATH, index=False)


def test_load_data():
    """Probar la carga de datos."""
    X, y = load_data(TEST_DATA_PATH)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == 10  # Ahora tenemos 10 muestras
    assert y.shape[0] == 10
    assert 'Outcome' not in X.columns
    assert y.name == 'Outcome'


def test_preprocess_and_train():
    """Probar el preprocesamiento y entrenamiento."""
    # Cargar datos
    X, y = load_data(TEST_DATA_PATH)

    # Preprocesar manualmente para evitar problemas con conjuntos pequeños
    features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Reemplazar ceros con NaN
    for feature in features_with_zeros:
        X.loc[X[feature] == 0, feature] = float('nan')

    # Imputar los valores NaN
    for feature in features_with_zeros:
        median_value = X[feature].median()
        X[feature] = X[feature].fillna(median_value)

    # División manual sin stratify para tests
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Verificar las formas de los conjuntos de datos
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]

    # Verificar que no hay valores faltantes
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()

    # Entrenar modelo
    model = train_model(X_train, y_train)

    # Verificar que el modelo es una pipeline con las etapas correctas
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps['scaler'], StandardScaler)
    assert isinstance(model.named_steps['classifier'], RandomForestClassifier)

    # Verificar que el modelo puede hacer predicciones
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)

    # Evaluar modelo
    metrics = evaluate_model(model, X_test, y_test)

    # Verificar que las métricas están presentes
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'confusion_matrix' in metrics


class TestDiabetesClassifier:
    """
    Pruebas para la clase DiabetesClassifier.
    """

    @pytest.fixture
    def model_path(self):
        """Crear un modelo de prueba y devolver su ruta."""
        # Entrenar un modelo simple con los datos de prueba
        X, y = load_data(TEST_DATA_PATH)

        # Preprocesar manualmente
        features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for feature in features_with_zeros:
            X.loc[X[feature] == 0, feature] = float('nan')
        for feature in features_with_zeros:
            median_value = X[feature].median()
            X[feature] = X[feature].fillna(median_value)

        # División manual
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = train_model(X_train, y_train)

        # Guardar el modelo en un archivo temporal
        model_path = os.path.join(TEST_DATA_DIR, "test_classifier_model.pkl")
        joblib.dump(model, model_path)

        yield model_path

        # Limpiar después de la prueba
        if os.path.exists(model_path):
            os.remove(model_path)

    def test_predict(self, model_path):
        """Probar la función de predicción."""
        classifier = DiabetesClassifier(model_path=model_path)

        # Caso de predicción válido
        features = {
            'Pregnancies': 6,
            'Glucose': 148,
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        }
        result = classifier.predict(features)

        # Verificar la estructura del resultado
        assert 'prediction' in result
        assert 'probability' in result
        assert 'diagnosis' in result

        # Verificar los tipos de datos
        assert isinstance(result['prediction'], int)
        assert isinstance(result['probability'], float)
        assert isinstance(result['diagnosis'], str)

        # Verificar que la predicción es 0 o 1
        assert result['prediction'] in [0, 1]

        # Verificar que la probabilidad está entre 0 y 1
        assert 0 <= result['probability'] <= 1