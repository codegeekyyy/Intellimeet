import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.database import engine
from app.config import settings
from app.routers import auth, audio, meetings

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

# Register routers
app.include_router(auth.router)
app.include_router(audio.router)
app.include_router(meetings.router)

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
