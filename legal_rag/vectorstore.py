"""Chroma vector store helpers."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .config import Settings


def get_vectorstore(settings: Settings, embeddings: Embeddings) -> Chroma:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
