# Usar una versión específica de Python como imagen base
FROM python:3.9-slim

# Definir variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requisitos y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente del proyecto y los datos
COPY app/ /app/app/
COPY data/ /app/data/
COPY models/ /app/models/
COPY setup.py .

# Instalar el proyecto como paquete (modo desarrollo)
RUN pip install -e .

# Exponer el puerto
EXPOSE $PORT

# Definir la variable de entorno para ejecución
ENV PYTHONPATH=/app

# Ejecutar la aplicación con Uvicorn
CMD ["python", "-m", "app.main"]