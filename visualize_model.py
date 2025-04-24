import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Configurar rutas
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
VISUALIZATIONS_DIR = os.path.join(ROOT_DIR, 'visualizations')

# Crear directorio de visualizaciones si no existe
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


def load_data_and_model(model_path=None, data_path=None):
    """
    Carga el modelo y los datos para visualización.

    Args:
        model_path (str): Ruta al archivo del modelo.
        data_path (str): Ruta al archivo de datos.

    Returns:
        tuple: (modelo, X, y) - modelo cargado y conjuntos de datos.
    """
    # Rutas por defecto
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, 'diabetes_model.pkl')
    if data_path is None:
        data_path = os.path.join(DATA_DIR, 'diabetes.csv')

    # Verificar si los archivos existen
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos en {data_path}")

    # Cargar el modelo
    print(f"Cargando modelo desde {model_path}...")
    model = joblib.load(model_path)

    # Cargar los datos
    print(f"Cargando datos desde {data_path}...")
    data = pd.read_csv(data_path)
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    return model, X, y


def visualize_feature_importance(model, X, output_path=None):
    """
    Visualiza la importancia de las características del modelo.

    Args:
        model: Modelo entrenado.
        X: DataFrame con las características.
        output_path (str): Ruta donde guardar la visualización.

    Returns:
        str: Ruta al archivo PNG generado.
    """
    # Extraer el clasificador de la pipeline
    classifier = model.named_steps['classifier']

    # Obtener la importancia de características
    feature_names = X.columns
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Crear DataFrame para seaborn
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })

    # Crear visualización
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Importancia de Características en el Modelo de Diabetes', fontsize=14)
    plt.xlabel('Importancia Relativa', fontsize=12)
    plt.ylabel('Característica', fontsize=12)
    plt.tight_layout()

    # Agregar valores de importancia en las barras
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        plt.text(width + 0.01, p.get_y() + p.get_height() / 2,
                 f'{width:.4f}', ha='left', va='center')

    # Guardar visualización
    if output_path is None:
        output_path = os.path.join(VISUALIZATIONS_DIR, 'feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualización de importancia de características guardada en {output_path}")
    return output_path


def visualize_tree(model, X, y, output_path=None):
    """
    Visualiza uno de los árboles del modelo (si es un conjunto de árboles).

    Args:
        model: Modelo entrenado.
        X: DataFrame con las características.
        y: Series con la variable objetivo.
        output_path (str): Ruta donde guardar la visualización.

    Returns:
        str: Ruta al archivo PNG generado.
    """
    # Extraer el clasificador de la pipeline
    classifier = model.named_steps['classifier']

    # Obtener nombres de características
    feature_names = X.columns.tolist()

    # Seleccionar el primer árbol para visualización
    plt.figure(figsize=(20, 10))
    try:
        tree.plot_tree(classifier.estimators_[0],
                       feature_names=feature_names,
                       class_names=['No Diabetes', 'Diabetes'],
                       filled=True,
                       rounded=True,
                       fontsize=10,
                       proportion=True)
        plt.title('Visualización de un Árbol del Modelo RandomForest', fontsize=16)
    except (AttributeError, IndexError):
        # Si no es un RandomForest o no tiene estimators_
        try:
            tree.plot_tree(classifier,
                           feature_names=feature_names,
                           class_names=['No Diabetes', 'Diabetes'],
                           filled=True,
                           rounded=True,
                           fontsize=10)
            plt.title('Visualización del Árbol de Decisión', fontsize=16)
        except:
            print("No se pudo visualizar el árbol. El modelo no parece ser un árbol de decisión.")
            return None

    # Guardar visualización
    if output_path is None:
        output_path = os.path.join(VISUALIZATIONS_DIR, 'tree_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualización del árbol guardada en {output_path}")
    return output_path


def visualize_confusion_matrix(model, X, y, output_path=None):
    """
    Visualiza la matriz de confusión del modelo.

    Args:
        model: Modelo entrenado.
        X: DataFrame con las características.
        y: Series con la variable objetivo.
        output_path (str): Ruta donde guardar la visualización.

    Returns:
        str: Ruta al archivo PNG generado.
    """
    # Realizar predicciones
    y_pred = model.predict(X)

    # Calcular matriz de confusión
    cm = confusion_matrix(y, y_pred)

    # Crear visualización
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Matriz de Confusión del Modelo de Diabetes', fontsize=14)
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)

    # Agregar texto con métricas
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    plt.figtext(0.5, 0.01,
                f'Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}',
                ha='center', fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Guardar visualización
    if output_path is None:
        output_path = os.path.join(VISUALIZATIONS_DIR, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Matriz de confusión guardada en {output_path}")
    return output_path


def visualize_roc_curve(model, X, y, output_path=None):
    """
    Visualiza la curva ROC del modelo.

    Args:
        model: Modelo entrenado.
        X: DataFrame con las características.
        y: Series con la variable objetivo.
        output_path (str): Ruta donde guardar la visualización.

    Returns:
        str: Ruta al archivo PNG generado.
    """
    # Obtener probabilidades para la clase positiva
    try:
        y_proba = model.predict_proba(X)[:, 1]
    except:
        print("El modelo no soporta predict_proba. No se puede generar la curva ROC.")
        return None

    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    # Crear visualización
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curva ROC del Modelo de Diabetes', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Guardar visualización
    if output_path is None:
        output_path = os.path.join(VISUALIZATIONS_DIR, 'roc_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Curva ROC guardada en {output_path}")
    return output_path


def visualize_model_summary(output_paths, output_path=None):
    """
    Crea una imagen resumen con todas las visualizaciones.

    Args:
        output_paths (list): Lista de rutas a las visualizaciones individuales.
        output_path (str): Ruta donde guardar la visualización resumen.

    Returns:
        str: Ruta al archivo PNG generado.
    """
    # Filtrar rutas None
    output_paths = [p for p in output_paths if p is not None]

    if not output_paths:
        return None

    # Determinar el número de filas y columnas para el grid
    n = len(output_paths)
    cols = min(2, n)
    rows = (n + cols - 1) // cols

    # Crear figura
    plt.figure(figsize=(15, 7 * rows))
    plt.suptitle('Resumen del Modelo de Predicción de Diabetes', fontsize=16, y=0.98)

    # Agregar cada imagen como un subplot
    for i, img_path in enumerate(output_paths):
        plt.subplot(rows, cols, i + 1)
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(os.path.basename(img_path).replace('.png', '').replace('_', ' ').title())
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Guardar resumen
    if output_path is None:
        output_path = os.path.join(VISUALIZATIONS_DIR, 'model_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Resumen del modelo guardado en {output_path}")
    return output_path


def main():
    """Función principal para la visualización del modelo."""
    print("Iniciando visualización del modelo de diabetes...")

    try:
        # Cargar modelo y datos
        model, X, y = load_data_and_model()

        # Generar visualizaciones
        output_paths = []
        output_paths.append(visualize_feature_importance(model, X))
        output_paths.append(visualize_tree(model, X, y))
        output_paths.append(visualize_confusion_matrix(model, X, y))
        output_paths.append(visualize_roc_curve(model, X, y))

        # Crear resumen
        visualize_model_summary(output_paths)

        print("\nVisualización completada exitosamente. Las imágenes se guardaron en el directorio 'visualizations/'.")

    except Exception as e:
        print(f"Error durante la visualización: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()