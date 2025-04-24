import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Configurar rutas
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Asegurar que el directorio de modelos existe
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(filepath=os.path.join(DATA_DIR, 'diabetes.csv')):
    """
    Cargar el conjunto de datos de diabetes.

    Args:
        filepath (str): Ruta al archivo CSV con los datos.

    Returns:
        X (pd.DataFrame): Características para el modelo.
        y (pd.Series): Variable objetivo (0: No diabetes, 1: Diabetes).
    """
    print(f"Cargando datos desde: {filepath}")
    # Cargar los datos
    data = pd.read_csv(filepath)

    # Mostrar información básica sobre los datos
    print(f"Forma del conjunto de datos: {data.shape}")
    print(f"Distribución de clases: {data['Outcome'].value_counts().to_dict()}")

    # Separar características y variable objetivo
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    return X, y


def preprocess_data(X, y):
    """
    Preprocesar los datos para el entrenamiento.

    Args:
        X (pd.DataFrame): Características.
        y (pd.Series): Variable objetivo.

    Returns:
        X_train, X_test, y_train, y_test: Conjuntos de entrenamiento y prueba.
    """
    # Manejar valores faltantes o cero en variables médicas
    # En este dataset, los ceros en ciertas columnas probablemente representan valores faltantes
    features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Reemplazar ceros con NaN (para poder identificarlos fácilmente)
    for feature in features_with_zeros:
        X.loc[X[feature] == 0, feature] = np.nan

    # Imputar los valores NaN con la mediana de cada columna
    for feature in features_with_zeros:
        median_value = X[feature].median()
        # Corregir la advertencia de fillna inplace
        X[feature] = X[feature].fillna(median_value)

    # Dividir en conjuntos de entrenamiento y prueba
    # Modificación para manejar conjuntos de datos pequeños
    if len(X) < 10:
        # Para conjuntos pequeños, usar una división simple sin estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=None
        )
    else:
        # Para conjuntos más grandes, usar la estratificación normal
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
    print(f"Tamaño del conjunto de prueba: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entrenar un modelo de clasificación para predecir diabetes.

    Args:
        X_train (pd.DataFrame): Características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.

    Returns:
        model (Pipeline): Modelo entrenado con pipeline de preprocesamiento.
    """
    print("Entrenando modelo de RandomForest...")

    # Crear un pipeline con escalado y modelo
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(model, X_test, y_test):
    """
    Evaluar el rendimiento del modelo.

    Args:
        model (Pipeline): Modelo entrenado.
        X_test (pd.DataFrame): Características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        dict: Métricas de rendimiento.
    """
    print("Evaluando modelo...")

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Mostrar resultados
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print(f"Precisión (Precision): {precision:.4f}")
    print(f"Sensibilidad (Recall): {recall:.4f}")
    print(f"Puntuación F1 (F1 Score): {f1:.4f}")
    print(f"Matriz de confusión:\n{conf_matrix}")

    # Guardar métricas en un diccionario
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist()
    }

    return metrics


def save_model(model, model_path=os.path.join(MODELS_DIR, 'diabetes_model.pkl')):
    """
    Guardar el modelo entrenado en disco.

    Args:
        model (Pipeline): Modelo entrenado a guardar.
        model_path (str): Ruta donde guardar el modelo.
    """
    print(f"Guardando modelo en: {model_path}")
    joblib.dump(model, model_path)
    print("Modelo guardado exitosamente.")


def main():
    """Función principal para el entrenamiento del modelo."""
    print("Iniciando proceso de entrenamiento del modelo de diabetes...")

    # Cargar los datos
    X, y = load_data()

    # Preprocesar los datos
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Entrenar el modelo
    model = train_model(X_train, y_train)

    # Evaluar el modelo
    metrics = evaluate_model(model, X_test, y_test)

    # Guardar el modelo
    save_model(model)

    print("Proceso de entrenamiento completado.")
    return metrics


if __name__ == "__main__":
    main()