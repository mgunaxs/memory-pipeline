"""
Main FastAPI application.

Entry point for the Memory Pipeline API with proper initialization,
middleware, error handling, and logging configuration.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings, create_directories
from app.core.database_prod import startup_database, shutdown_database, init_database
from app.core.chromadb_prod import startup_chromadb, shutdown_chromadb
from app.api.endpoints import memory

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('memory_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown tasks including database initialization
    and directory creation.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting Memory Pipeline API")

    try:
        # Create necessary directories
        create_directories()
        logger.info("Created storage directories")

        # Initialize PostgreSQL database
        await startup_database()
        logger.info("PostgreSQL database connected successfully")

        # Initialize ChromaDB cloud
        await startup_chromadb()
        logger.info("ChromaDB cloud connected successfully")

        logger.info(f"Memory Pipeline API started successfully on {settings.api_version}")

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Memory Pipeline API")

    try:
        # Close database connections
        await shutdown_database()
        logger.info("Database connections closed")

        # Close ChromaDB connections
        await shutdown_chromadb()
        logger.info("ChromaDB connections closed")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware with secure configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

    return response


@app.middleware("http")
async def request_size_limit_middleware(request: Request, call_next):
    """Limit request body size to prevent DoS attacks."""
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB

    if request.headers.get("content-length"):
        content_length = int(request.headers["content-length"])
        if content_length > MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": "Request too large", "max_size": "10MB"}
            )

    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Request logging middleware.

    Logs all incoming requests with timing information.

    Args:
        request: FastAPI request
        call_next: Next middleware in chain

    Returns:
        Response: HTTP response
    """
    import time

    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Log request details
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)

    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTP exception handler.

    Provides consistent error response format.

    Args:
        request: FastAPI request
        exc: HTTP exception

    Returns:
        JSONResponse: Standardized error response
    """
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": exc.detail,
            "timestamp": "2024-01-01T00:00:00Z"  # Will be overridden by actual timestamp
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    General exception handler.

    Handles unexpected errors with proper logging.

    Args:
        request: FastAPI request
        exc: Exception

    Returns:
        JSONResponse: Error response
    """
    logger.error(f"Unhandled exception: {exc} - {request.url}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": "Internal server error",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    )


# Include routers
app.include_router(memory.router, prefix="/api/v1")


@app.get("/")
async def root():
    """
    Root endpoint with API information.

    Returns:
        Dict: API information and status
    """
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/memory/health",
            "extract": "/api/v1/memory/extract",
            "search": "/api/v1/memory/search"
        }
    }


@app.get("/api/v1/info")
async def api_info():
    """
    API information endpoint.

    Returns:
        Dict: Detailed API information
    """
    return {
        "api": {
            "title": settings.api_title,
            "version": settings.api_version,
            "description": settings.api_description
        },
        "features": [
            "Memory extraction from natural language",
            "Semantic search with vector embeddings",
            "Multiple memory types (fact, preference, event, routine, emotion)",
            "Automatic expiration and importance scoring",
            "Rate limiting and caching",
            "Health monitoring"
        ],
        "models": {
            "llm": "Google Gemini 1.5 Flash",
            "embeddings": "SentenceTransformers all-MiniLM-L6-v2",
            "vector_db": "ChromaDB Cloud",
            "database": "PostgreSQL (AWS RDS)"
        },
        "limits": {
            "max_text_length": settings.max_text_length,
            "max_memories_per_user": settings.max_memories_per_user,
            "rate_limit_per_minute": settings.rate_limit_per_minute
        }
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Memory Pipeline API server")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )