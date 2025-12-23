import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration for the pipeline."""

    # API & LLM Settings
    OPENAI_API_URL: str = os.getenv("OPENAI_API_URL", "http://localhost:8000/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "dummy")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "qwen3-235b-fp8")

    # Training & Inference Settings
    MAX_SEQ_LENGTH: int = 2048

    # Project Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_DIR: Path = DATA_DIR / "chroma"
    DB_PATH: str = str(DATA_DIR / "aetheris.db")

    # Prompt Template (Single Source of Truth)
    PROMPT_STYLE: str = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "You are an expert in AuraDSL. Translate the natural language request "
        "into a valid AuraDSL query based on the provided schema.\n\n"
        "### Context:\n"
        "{}\n\n"
        "### Input:\n"
        "{}\n\n"
        "### Response:\n"
        "{}"
    )

    @classmethod
    def ensure_dirs(cls) -> None:
        """Creates necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_DIR.mkdir(parents=True, exist_ok=True)


Config.ensure_dirs()
