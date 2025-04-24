import os
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.model.diabetes_classifier import DiabetesClassifier
from app.model.train import load_data, preprocess_data, train_model, evaluate_model

# Ruta al directorio de datos de prueba
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Datos de prueba simples
TEST_DATA = pd.DataFrame({
    'Pregnancies': [1, 8, 1, 0, 5],
    'Glucose': [89, 183, 89, 137, 116],
    'BloodPressure': [66, 64, 66, 40, 74],
    'SkinThickness': [23, 0, 94, 35, 0],
    'Insulin': [94, 0, 0, 168, 0],
    'BMI': [28.1, 23.3, 28.1, 43.1, 25.6],
    'DiabetesPedigreeFunction': [0.167, 0.672, 0.167, 2.288, 0.201],
    'Age': [21, 32, 21, 33, 30],
    'Outcome': [0, 1, 0, 1, 0]
})

# Guardar datos de prueba en un archivo CSV
TEST_DATA_PATH = os.path.join(TEST_DATA_DIR, "test_diabetes.csv")
TEST_DATA.to_csv(TEST_DATA_PATH, index=False)


class TestTraining:
    """
    Pruebas para las funciones de entrenamiento.
    """

    def test_load_data(self):
        """Probar la carga de datos."""
        X, y = load_data(TEST_DATA_PATH)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == 5
        assert y.shape[0] == 5
        assert 'Outcome' not in X.columns
        assert y.name == 'Outcome'

    def test_preprocess_data(self):
        """Probar el preprocesamiento de datos."""
        X, y = load_data(TEST_DATA_PATH)
        X_train, X_test, y_train, y_test = preprocess_data(X, y)

        # Verificar las formas de los conjuntos de datos
        assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
        assert y_train.shape[0] + y_test.shape[0] == y.shape[0]

        # Verificar que no hay valores faltantes
        assert not X_train.isna().any().any()
        assert not X_test.isna().any().any()

    def test_train_model(self):
        """Probar el entrenamiento del modelo."""
        X, y = load_data(TEST_DATA_PATH)
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        model = train_model(X_train, y_train)

        # Verificar que el modelo es una pipeline con las etapas correctas
        assert isinstance(model, Pipeline)
        assert isinstance(model.named_steps['scaler'], StandardScaler)
        assert isinstance(model.named_steps['classifier'], RandomForestClassifier)

        # Verificar que el modelo puede hacer predicciones
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

    def test_evaluate_model(self):
        """Probar la evaluación del modelo."""
        X, y = load_data(TEST_DATA_PATH)
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # Verificar que las métricas están presentes
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics

        # Verificar que las métricas tienen valores válidos
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1


class TestDiabetesClassifier:
    """
    Pruebas para la clase DiabetesClassifier.
    """

    @pytest.fixture
    def model_path(self):
        """Crear un modelo de prueba y devolver su ruta."""
        # Entrenar un modelo simple con los datos de prueba
        X, y = load_data(TEST_DATA_PATH)
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        model = train_model(X_train, y_train)

        # Guardar el modelo en un archivo temporal
        model_path = os.path.join(TEST_DATA_DIR, "test_model.pkl")
        joblib.dump(model, model_path)

        yield model_path

        # Limpiar después de la prueba
        if os.path.exists(model_path):
            os.remove(model_path)

    def test_init(self, model_path):
        """Probar la inicialización del clasificador."""
        classifier = DiabetesClassifier(model_path=model_path)
        assert classifier.model is not None
        assert isinstance(classifier.model, Pipeline)

    def test_validate_features(self, model_path):
        """Probar la validación de características."""
        classifier = DiabetesClassifier(model_path=model_path)

        # Caso válido
        valid_features = {
            'Pregnancies': 6,
            'Glucose': 148,
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        }
        assert classifier._validate_features(valid_features) is True

        # Caso con características faltantes
        invalid_features = {
            'Pregnancies': 6,
            'Glucose': 148,
            # Falta BloodPressure
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        }
        with pytest.raises(ValueError):
            classifier._validate_features(invalid_features)

        # Caso con características adicionales
        invalid_features = {
            'Pregnancies': 6,
            'Glucose': 148,
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50,
            'ExtraFeature': 100  # Característica adicional
        }
        with pytest.raises(ValueError):
            classifier._validate_features(invalid_features)

    def test_preprocess_input(self, model_path):
        """Probar el preprocesamiento de la entrada."""
        classifier = DiabetesClassifier(model_path=model_path)

        # Caso válido
        valid_features = {
            'Pregnancies': 6,
            'Glucose': 148,
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        }
        df = classifier._preprocess_input(valid_features)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 8)
        assert list(df.columns) == classifier.FEATURE_NAMES

        # Caso con valor no numérico
        invalid_features = {
            'Pregnancies': 6,
            'Glucose': 'not_a_number',  # Valor no numérico
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        }
        with pytest.raises(ValueError):
            classifier._preprocess_input(invalid_features)

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

        # Verificar que el diagnóstico coincide con la predicción
        if result['prediction'] == 1:
            assert result['diagnosis'] == 'Diabetes'
        else:
            assert result['diagnosis'] == 'No Diabetes'

    def test_batch_predict(self, model_path):
        """Probar la función de predicción por lotes."""
        classifier = DiabetesClassifier(model_path=model_path)

        # Lista de características para predicción por lotes
        features_list = [
            {
                'Pregnancies': 6,
                'Glucose': 148,
                'BloodPressure': 72,
                'SkinThickness': 35,
                'Insulin': 0,
                'BMI': 33.6,
                'DiabetesPedigreeFunction': 0.627,
                'Age': 50
            },
            {
                'Pregnancies': 1,
                'Glucose': 85,
                'BloodPressure': 66,
                'SkinThickness': 29,
                'Insulin': 0,
                'BMI': 26.6,
                'DiabetesPedigreeFunction': 0.351,
                'Age': 31
            },
            # Caso inválido que debería generar un error
            {
                'Pregnancies': 8,
                'Glucose': 183,
                # Falta BloodPressure
                'SkinThickness': 0,
                'Insulin': 0,
                'BMI': 23.3,
                'DiabetesPedigreeFunction': 0.672,
                'Age': 32
            }
        ]

        results = classifier.batch_predict(features_list)

        # Verificar que hay 3 resultados
        assert len(results) == 3

        # Verificar que los dos primeros resultados son predicciones válidas
        assert 'prediction' in results[0]
        assert 'probability' in results[0]
        assert 'diagnosis' in results[0]

        assert 'prediction' in results[1]
        assert 'probability' in results[1]
        assert 'diagnosis' in results[1]

        # Verificar que el tercer resultado contiene un error
        assert 'error' in results[2]