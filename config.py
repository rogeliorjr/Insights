"""Standalone configuration for insights analysis"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


def _load_env_file():
    """Load environment variables from .env file if it exists"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value


# Load .env file on module import
_load_env_file()


@dataclass
class StandaloneSettings:
    """Settings for standalone insights module - loaded from environment variables"""

    # Database
    DATABASE_URL: str = ""

    # AI APIs
    OPENAI_API_KEY: Optional[str] = None
    LLAMAINDEX_API_KEY: Optional[str] = None

    # Vector Store Settings
    VECTOR_INSERT_BATCH_SIZE: int = 50

    def __post_init__(self):
        """Load settings from environment variables"""
        self.DATABASE_URL = os.getenv(
            "DATABASE_URL",
            os.getenv("STANDALONE_DATABASE_URL", "postgresql://user:password@localhost/insightsync_db")
        )
        # Fix Heroku DATABASE_URL format
        if self.DATABASE_URL.startswith("postgres://"):
            self.DATABASE_URL = self.DATABASE_URL.replace("postgres://", "postgresql://", 1)

        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.LLAMAINDEX_API_KEY = os.getenv("LLAMAINDEX_API_KEY")
        self.VECTOR_INSERT_BATCH_SIZE = int(os.getenv("VECTOR_INSERT_BATCH_SIZE", "50"))

    def validate_for_analysis(self):
        """Validate settings required for running analysis (call before using LlamaIndex)"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for analysis")


# Singleton settings instance
settings = StandaloneSettings()