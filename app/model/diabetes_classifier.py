import os
import joblib
import numpy as np
import pandas as pd


class DiabetesClassifier:
    """
    Clase para predecir la probabilidad de diabetes utilizando el modelo entrenado.
    """

    # Nombres de las características esperadas por el modelo
    FEATURE_NAMES = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    def __init__(self, model_path=None):
        """
        Inicializar el clasificador de diabetes.

        Args:
            model_path (str, optional): Ruta al archivo del modelo serializado.
                Si no se proporciona, se usa la ruta por defecto.
        """
        if model_path is None:
            # Ruta por defecto relativa a la ubicación de este archivo
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, 'models', 'diabetes_model.pkl')

        # Cargar el modelo
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Cargar el modelo desde un archivo.

        Args:
            model_path (str): Ruta al archivo del modelo serializado.

        Returns:
            object: Modelo cargado.

        Raises:
            FileNotFoundError: Si el archivo del modelo no existe.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {model_path}")

        return joblib.load(model_path)

    def _validate_features(self, features):
        """
        Validar que se proporcionen todas las características necesarias.

        Args:
            features (dict): Diccionario con las características.

        Returns:
            bool: True si todas las características están presentes.

        Raises:
            ValueError: Si faltan características o hay características adicionales.
        """
        feature_set = set(features.keys())
        expected_set = set(self.FEATURE_NAMES)

        missing = expected_set - feature_set
        additional = feature_set - expected_set

        if missing:
            raise ValueError(f"Faltan las siguientes características: {missing}")

        if additional:
            raise ValueError(f"Se proporcionaron características adicionales no esperadas: {additional}")

        return True

    def _preprocess_input(self, features):
        """
        Preprocesar las características de entrada para asegurar que están en el formato correcto.

        Args:
            features (dict): Diccionario con las características.

        Returns:
            pd.DataFrame: DataFrame con las características procesadas.
        """
        # Convertir a DataFrame para mantener el orden de las características
        df = pd.DataFrame([features])[self.FEATURE_NAMES]

        # Validar tipos de datos y convertir si es necesario
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                raise ValueError(f"El valor para '{col}' no es numérico: {df[col][0]}")

        return df

    def predict(self, features):
        """
        Predecir la probabilidad de diabetes para un conjunto de características.

        Args:
            features (dict): Diccionario con las características del paciente.

        Returns:
            dict: Diccionario con la clase predicha y la probabilidad.
        """
        # Validar las características
        self._validate_features(features)

        # Preprocesar la entrada
        X = self._preprocess_input(features)

        # Hacer la predicción
        prediction = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0][1])

        return {
            'prediction': prediction,
            'probability': probability,
            'diagnosis': 'Diabetes' if prediction == 1 else 'No Diabetes'
        }

    def batch_predict(self, features_list):
        """
        Predecir la probabilidad de diabetes para múltiples conjuntos de características.

        Args:
            features_list (list): Lista de diccionarios con las características.

        Returns:
            list: Lista de predicciones.
        """
        results = []
        for features in features_list:
            try:
                prediction = self.predict(features)
                results.append(prediction)
            except Exception as e:
                results.append({'error': str(e)})

        return results