"""Runtime configuration, read from environment variables / a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq").lower())
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.2"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    hf_model: str = field(default_factory=lambda: _env("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))

    # Embeddings
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # Vector store
    chroma_dir: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))))
    chroma_collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "indian_legal_kb"))

    # Ingestion / retrieval
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))


def get_settings() -> Settings:
    return Settings()
