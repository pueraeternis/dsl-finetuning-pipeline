import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


class Config:
    """Centralized configuration for the pipeline."""

    # API & LLM Settings
    OPENAI_API_URL: str = os.getenv("OPENAI_API_URL", "http://localhost:8000/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "dummy")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "qwen3-235b-fp8")

    # Project Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_DIR: Path = DATA_DIR / "chroma"
    DB_PATH: str = str(DATA_DIR / "aetheris.db")

    # Dataset Generation Settings
    SAMPLES_TO_GENERATE: int = 10000
    BATCH_SIZE: int = 10

    @classmethod
    def ensure_dirs(cls) -> None:
        """Creates necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# Pre-create directories
Config.ensure_dirs()
