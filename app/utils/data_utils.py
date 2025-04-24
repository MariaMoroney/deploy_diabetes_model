import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional


def validate_csv_format(file_path: str) -> bool:
    """
    Valida que un archivo CSV tenga el formato correcto para el modelo de diabetes.

    Args:
        file_path (str): Ruta al archivo CSV a validar.

    Returns:
        bool: True si el formato es válido, False en caso contrario.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")

    # Columnas requeridas para el modelo de diabetes
    required_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    # Intentar leer solo las cabeceras del CSV
    try:
        headers = pd.read_csv(file_path, nrows=0).columns.tolist()

        # Verificar si todas las columnas requeridas están presentes
        for column in required_columns:
            if column not in headers:
                return False

        return True
    except Exception as e:
        print(f"Error al validar el formato del CSV: {e}")
        return False


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y preprocesa un DataFrame de datos de diabetes.

    Args:
        df (pd.DataFrame): DataFrame con datos a limpiar.

    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    # Crear una copia para no modificar el original
    df_clean = df.copy()

    # Columnas que no deberían tener valores de cero
    zero_invalid_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Reemplazar ceros con NaN
    for column in zero_invalid_columns:
        if column in df_clean.columns:
            df_clean[column] = df_clean[column].replace(0, np.nan)

    # Imputar valores faltantes con la mediana
    for column in zero_invalid_columns:
        if column in df_clean.columns:
            median_value = df_clean[column].median()
            df_clean[column] = df_clean[column].fillna(median_value)

    return df_clean


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Genera estadísticas descriptivas para un DataFrame de datos de diabetes.

    Args:
        df (pd.DataFrame): DataFrame con datos.

    Returns:
        Dict[str, Dict[str, float]]: Diccionario con estadísticas por columna.
    """
    stats = {}

    for column in df.columns:
        if df[column].dtype in [np.int64, np.float64]:
            stats[column] = {
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'mean': float(df[column].mean()),
                'median': float(df[column].median()),
                'std': float(df[column].std())
            }

    return stats


def split_features_target(df: pd.DataFrame, target_column: str = 'Outcome') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Divide un DataFrame en características y variable objetivo.

    Args:
        df (pd.DataFrame): DataFrame completo.
        target_column (str): Nombre de la columna objetivo.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.Series]]: Tupla con características y variable objetivo.
    """
    if target_column in df.columns:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        return X, y
    else:
        # Si no hay columna objetivo, devolver solo las características
        return df, None


def format_predictions_for_export(predictions: List[Dict[str, Any]], include_features: bool = False,
                                  features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Formatea las predicciones para exportar a CSV.

    Args:
        predictions (List[Dict[str, Any]]): Lista de predicciones.
        include_features (bool): Si se incluyen las características originales.
        features (Optional[pd.DataFrame]): DataFrame con las características originales.

    Returns:
        pd.DataFrame: DataFrame con las predicciones formateadas.
    """
    # Crear DataFrame con las predicciones
    pred_df = pd.DataFrame(predictions)

    if include_features and features is not None:
        # Unir las características con las predicciones
        return pd.concat([features.reset_index(drop=True), pred_df], axis=1)

    return pred_df