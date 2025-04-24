# API de Predicción de Diabetes (MLOps)

Un sistema completo de MLOps para predecir diabetes utilizando características médicas.

**Autora:** María Fernanda Moroney Sole

## Descripción del Proyecto

Este proyecto implementa un sistema de MLOps completo para un modelo de predicción de diabetes basado en características médicas de pacientes, utilizando el conjunto de datos Pima Indians Diabetes. El sistema incluye:

- Un modelo de machine learning (RandomForest) para la predicción de diabetes
- Una API REST construida con FastAPI para consumir el modelo
- Un sistema de CI/CD con GitHub Actions para automatizar pruebas, entrenamiento y despliegue
- Dockerización para garantizar la portabilidad y reproducibilidad
- Visualizaciones interactivas del modelo y su rendimiento

## Estructura del Proyecto

```
diabetes-ml-model/
├── .github/workflows/     # Flujos de trabajo de CI/CD
│   ├── develop.yml        # CI para la rama de desarrollo
│   ├── staging.yml        # CI/CD para la rama de staging
│   └── main.yml           # CD para la rama de producción
├── app/                   # Código fuente de la aplicación
│   ├── api/               # Definición de los endpoints
│   ├── model/             # Código del modelo
│   └── utils/             # Utilidades para manejo de datos
├── data/                  # Datos para entrenamiento
├── models/                # Modelos serializados
├── tests/                 # Pruebas unitarias
├── visualizations/        # Visualizaciones del modelo
├── Dockerfile             # Configuración para Docker
├── docker-compose.yml     # Configuración para Docker Compose
├── visualize_model.py     # Script para visualizar el modelo
└── README.md              # Documentación
```

## Características Técnicas

- **Modelo:** RandomForest con preprocesamiento usando StandardScaler
- **Precisión:** ~75-80% en el conjunto de prueba
- **API:** FastAPI con validación y documentación automática
- **Automatización:** GitHub Actions para CI/CD
- **Containerización:** Docker para garantizar la portabilidad
- **Visualización:** Gráficos de importancia de características, árboles de decisión, matriz de confusión y curva ROC

## Variables de Entrada

El modelo toma las siguientes variables para la predicción:

1. **Pregnancies**: Número de embarazos
2. **Glucose**: Concentración de glucosa en plasma (mg/dL)
3. **BloodPressure**: Presión arterial diastólica (mm Hg)
4. **SkinThickness**: Grosor del pliegue cutáneo del tríceps (mm)
5. **Insulin**: Insulina sérica de 2 horas (mu U/ml)
6. **BMI**: Índice de masa corporal (peso en kg/(altura en m)^2)
7. **DiabetesPedigreeFunction**: Función de pedigrí de diabetes
8. **Age**: Edad (años)

## Variable de Salida

- **Outcome**: 0 (No Diabetes) o 1 (Diabetes)
- **Probabilidad**: Valor entre 0 y 1 que indica la probabilidad de diabetes
- **Diagnóstico**: Interpretación en lenguaje natural ("Diabetes" o "No Diabetes")

## Instalación y Uso

### Requisitos

- Python 3.8+
- Docker y Docker Compose
- Git

### Configuración Local

```bash
# Clonar el repositorio
git clone https://github.com/MariaMoroney/deploy_diabetes_model.git
cd deploy_diabetes_model

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo (si no existe)
python -m app.model.train

# Visualizar el modelo
python visualize_model.py

# Ejecutar la API
python -m app.main
```

### Uso con Docker

```bash
# Construir la imagen
docker build -t diabetes-api .

# Ejecutar el contenedor
docker run -p 8000:8000 diabetes-api
```

La API estará disponible en http://localhost:8000 y la documentación interactiva en http://localhost:8000/docs.

## Flujo de Trabajo de CI/CD

El proyecto implementa un flujo de trabajo MLOps completo con tres ramas:

1. **develop**: Para desarrollo y pruebas continuas
   - Ejecuta pruebas automatizadas
   - Entrena el modelo automáticamente

2. **staging**: Para pruebas de integración
   - Construye y publica la imagen Docker con la etiqueta "staging"
   - Simula un despliegue en un entorno de pruebas

3. **main**: Para producción
   - Construye y publica la imagen Docker con la etiqueta "latest"
   - Despliega en el entorno de producción

## Ejemplo de Uso de la API

### Predicción Individual

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "Pregnancies": 6,
         "Glucose": 148,
         "BloodPressure": 72,
         "SkinThickness": 35,
         "Insulin": 0,
         "BMI": 33.6,
         "DiabetesPedigreeFunction": 0.627,
         "Age": 50
     }'
```

### Predicción por Lotes

```bash
curl -X POST "http://localhost:8000/api/v1/batch-predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

## Visualizaciones

El proyecto incluye visualizaciones generadas automáticamente para facilitar la interpretación del modelo:

- **Importancia de Características**: Muestra qué variables tienen mayor impacto en las predicciones
- **Visualización de Árbol**: Ilustra cómo el modelo toma decisiones
- **Matriz de Confusión**: Evalúa el rendimiento del modelo
- **Curva ROC**: Muestra la calidad del clasificador a diferentes umbrales

Para generar estas visualizaciones, ejecute:

```bash
python visualize_model.py
```

Las imágenes se guardarán en el directorio `visualizations/`.

