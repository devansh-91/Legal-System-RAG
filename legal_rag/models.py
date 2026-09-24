"""Factories for the free LLMs and embedding models used by the app."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings

SUPPORTED_PROVIDERS = ("groq", "ollama", "huggingface")


def get_embeddings(settings: Settings) -> Embeddings:
    """Local sentence-transformers embeddings — free and runs on CPU."""
    from langchain_huggingface import HuggingFaceEmbeddings

    model = settings.embedding_model
    encode_kwargs = {"normalize_embeddings": True}
    # E5 models expect "query: " / "passage: " prefixes; BGE uses a query instruction.
    if "e5" in model.lower():
        return HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={**encode_kwargs, "prompt": "passage: "},
            query_encode_kwargs={**encode_kwargs, "prompt": "query: "},
        )
    if "bge" in model.lower() and "-en" in model.lower():
        return HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs=encode_kwargs,
            query_encode_kwargs={
                **encode_kwargs,
                "prompt": "Represent this sentence for searching relevant passages: ",
            },
        )
    return HuggingFaceEmbeddings(model_name=model, encode_kwargs=encode_kwargs)


def get_llm(settings: Settings) -> BaseChatModel:
    """Return a chat model for the configured free provider."""
    provider = settings.llm_provider

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.groq_model, temperature=settings.llm_temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=settings.hf_model,
            task="text-generation",
            temperature=max(settings.llm_temperature, 0.01),
            max_new_tokens=1024,
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(
        f"Unknown LLM_PROVIDER {provider!r}; choose one of {', '.join(SUPPORTED_PROVIDERS)}"
    )
