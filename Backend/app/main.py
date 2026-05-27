import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.database import engine
from app.config import settings
from app.routers import auth, audio, meetings, interview

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("intellimeet")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Intellimeet API server...")
    logger.info("Verifying database connection...")
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection verified successfully! [OK]")
    except Exception as e:
        logger.critical(f"Database connection verification failed: {e} [FAILED]")
        raise e
    yield
    logger.info("Shutting down API server...")
    await engine.dispose()
    logger.info("Database engine resources cleaned up.")

# Initialize FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    description="Meeting Summarizer & Interview Prep Platform API",
    version="1.0.0",
    lifespan=lifespan,
    debug=settings.DEBUG
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "error_code": "VALIDATION_ERROR",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    detail = str(exc) if settings.DEBUG else "An unexpected internal server error occurred."
    return JSONResponse(
        status_code=500,
        content={
            "detail": detail,
            "error_code": "INTERNAL_SERVER_ERROR",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

# Register routers
app.include_router(auth.router)
app.include_router(audio.router)
app.include_router(meetings.router)
app.include_router(interview.router)

# Root route
@app.get("/")
async def read_root():
    return {"message": f"Welcome to {settings.APP_NAME} API!"}

# Health check route
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "version": "1.0.0"
    }
