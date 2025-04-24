import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.endpoints import router as api_router

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Diabetes",
    description="API para predecir la probabilidad de diabetes basado en características médicas",
    version="1.0.0",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, esto debe ser restringido
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir el router de la API
app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """
    Manejador global de excepciones.

    Args:
        request: Solicitud que causó la excepción.
        exc (Exception): Excepción ocurrida.

    Returns:
        JSONResponse: Respuesta JSON con el error.
    """
    logger.error(f"Error no manejado: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor. Por favor, contacte al administrador."}
    )


@app.get("/")
async def root():
    """
    Endpoint raíz de la aplicación.

    Returns:
        dict: Mensaje de bienvenida.
    """
    return {
        "message": "Bienvenido a la API de Predicción de Diabetes",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health():
    """
    Endpoint para verificar la salud de la aplicación.

    Returns:
        dict: Estado de la aplicación.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Obtener el puerto de la variable de entorno o usar 8000 por defecto
    port = int(os.getenv("PORT", 8000))

    # Iniciar el servidor
    logger.info(f"Iniciando servidor en el puerto {port}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)