"""
FastAPI application for serving multilingual-e5-small embeddings.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from models import EmbeddingRequest, EmbeddingResponse, HealthResponse, ErrorResponse
from embedding_service import embedding_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles model loading on startup and cleanup on shutdown.
    """
    # Startup
    print("🚀 Starting up FastAPI application...")
    try:
        embedding_service.load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

    yield

    # Shutdown
    print("🛑 Shutting down FastAPI application...")
    embedding_service.unload_model()
    print("✅ Cleanup completed")


# Create FastAPI application
app = FastAPI(
    title="Multilingual E5 Small Embedding API",
    description="REST API for generating embeddings using intfloat/multilingual-e5-small model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Multilingual E5 Small Embedding API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        is_loaded = embedding_service.is_loaded()
        return HealthResponse(
            status="healthy" if is_loaded else "unhealthy",
            model_name=embedding_service.model_name,
            model_loaded=is_loaded
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {e}"
        )


@app.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings from tokenized input.

    This endpoint expects input that has already been tokenized using the
    multilingual-e5-small tokenizer.
    """
    try:
        if not embedding_service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded"
            )

        # Generate embeddings
        embeddings, shape = embedding_service.get_embeddings(
            request.input_ids,
            request.attention_mask
        )

        return EmbeddingResponse(
            embedding=embeddings,
            shape=shape
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Request processing failed: {e}"
        )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if not embedding_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )

    return {
        "model_name": embedding_service.model_name,
        "device": embedding_service.device,
        "status": "loaded"
    }


if __name__ == "__main__":
    import uvicorn

    # Get server configuration
    server_config = settings.get('SERVER', settings.get('server', {}))
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 8000)
    reload = server_config.get('reload', False)

    print(f"🌐 Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload
    )