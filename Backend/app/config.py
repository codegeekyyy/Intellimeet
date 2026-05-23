import os 
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Intellimeet"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_PORT: int = 8000

    # Database & Redis
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"

    # Authentication
    JWT_SECRET_KEY: str
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Audio Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 500

    # ML/AI Config
    WHISPER_MODEL: str = "base"
    USE_GPU: bool = False
    HUGGINGFACE_TOKEN: str
    GROQ_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
