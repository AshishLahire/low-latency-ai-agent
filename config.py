# =============================================
# config.py - Centralized configuration
# =============================================
# This file loads all environment variables
# and provides type-safe configuration

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator, Field
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Supabase
    SUPABASE_URL: str = Field(..., env='SUPABASE_URL')
    SUPABASE_KEY: str = Field(..., env='SUPABASE_KEY')
    SUPABASE_JWT_SECRET: Optional[str] = Field(None, env='SUPABASE_JWT_SECRET')
    
    # Groq
    GROQ_API_KEY: str = Field(..., env='GROQ_API_KEY')
    
    # Redis
    REDIS_URL: str = Field('redis://localhost:6379', env='REDIS_URL')
    
    # App Settings
    MAX_HISTORY_LENGTH: int = Field(20, env='MAX_HISTORY_LENGTH')
    RAG_MATCH_THRESHOLD: float = Field(0.78, env='RAG_MATCH_THRESHOLD')
    RAG_MATCH_COUNT: int = Field(5, env='RAG_MATCH_COUNT')
    RATE_LIMIT_MESSAGES: int = Field(30, env='RATE_LIMIT_MESSAGES')
    RATE_LIMIT_WINDOW: int = Field(60, env='RATE_LIMIT_WINDOW')
    
    # Model Settings
    EMBEDDING_MODEL: str = Field('all-mpnet-base-v2', env='EMBEDDING_MODEL')
    LLM_MODEL: str = Field('mixtral-8x7b-32768', env='LLM_MODEL')
    
    # Server
    PORT: int = Field(8000, env='PORT')
    ENVIRONMENT: str = Field('development', env='ENVIRONMENT')
    CORS_ORIGINS: List[str] = Field(['*'], env='CORS_ORIGINS')
    
    # Security
    SECRET_KEY: str = Field('dev-secret-key-change-in-prod', env='SECRET_KEY')
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env='ACCESS_TOKEN_EXPIRE_MINUTES')
    
    # Logging
    LOG_LEVEL: str = Field('INFO', env='LOG_LEVEL')
    
    @validator('CORS_ORIGINS', pre=True)
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        """Validate environment value"""
        if v not in ['development', 'staging', 'production']:
            raise ValueError('ENVIRONMENT must be development, staging, or production')
        return v
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """Validate log level"""
        if v not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            raise ValueError('LOG_LEVEL must be DEBUG, INFO, WARNING, or ERROR')
        return v
    
    class Config:
        env_file = '.env'
        case_sensitive = True

# Create global settings object
settings = Settings()

# Validate required settings on import
def validate_settings():
    """Validate that all required settings are present"""
    required = ['SUPABASE_URL', 'SUPABASE_KEY', 'GROQ_API_KEY']
    missing = [req for req in required if not getattr(settings, req)]
    
    if missing:
        raise ValueError(
            f"Missing required environment variables: {missing}\n"
            "Please check your .env file"
        )
    
    # Print config summary (without secrets)
    print(f"""
    ========================================
    HelpKart AI Agent Configuration
    ========================================
    Environment: {settings.ENVIRONMENT}
    Port: {settings.PORT}
    Model: {settings.LLM_MODEL}
    Max History: {settings.MAX_HISTORY_LENGTH}
    RAG Threshold: {settings.RAG_MATCH_THRESHOLD}
    RAG Count: {settings.RAG_MATCH_COUNT}
    Rate Limit: {settings.RATE_LIMIT_MESSAGES}/{settings.RATE_LIMIT_WINDOW}s
    CORS Origins: {settings.CORS_ORIGINS}
    Log Level: {settings.LOG_LEVEL}
    ========================================
    """)

# Run validation
validate_settings()
