from __future__ import annotations

import dataclasses

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from legal_rag.config import PROJECT_ROOT, Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return dataclasses.replace(
        Settings(),
        chroma_dir=tmp_path / "chroma",
        chroma_collection="test_kb",
        data_dir=PROJECT_ROOT / "data",
        top_k=3,
    )


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=64)
