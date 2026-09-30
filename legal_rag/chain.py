"""The retrieval-augmented generation pipeline.

Flow:  (history + question) -> condense to standalone question -> retrieve from Chroma
       -> build numbered context -> LLM answer with inline citations + sources list.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore

from .config import Settings, get_settings
from .prompts import CONDENSE_PROMPT, DISCLAIMER, QA_PROMPT
from .safety import detect_urgency

# Only keep the last few turns to stay within free-tier context limits.
MAX_HISTORY_MESSAGES = 6


@dataclass
class Source:
    id: int
    title: str
    section: str
    source: str
    reference: str
    snippet: str
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def to_messages(history: Sequence[dict[str, str]] | None) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in (history or [])[-MAX_HISTORY_MESSAGES:]:
        content = turn.get("content", "")
        if turn.get("role") == "user":
            messages.append(HumanMessage(content))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content))
    return messages


def format_context(docs: Sequence[Document]) -> str:
    if not docs:
        return "(no relevant passages found)"
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        label = meta.get("title", "Document")
        if meta.get("page"):
            label += f", page {meta['page']}"
        blocks.append(f"[{i}] ({label})\n{doc.page_content}")
    return "\n\n".join(blocks)


def to_sources(docs: Sequence[Document]) -> list[Source]:
    sources = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        text = doc.page_content
        if text.startswith("[") and "]\n" in text:
            text = text.split("]\n", 1)[1]
        sources.append(
            Source(
                id=i,
                title=meta.get("title", ""),
                section=meta.get("section", ""),
                source=meta.get("source", ""),
                reference=meta.get("reference", ""),
                snippet=text[:400].strip(),
                page=meta.get("page"),
            )
        )
    return sources


class LegalRAG:
    def __init__(
        self,
        llm: BaseChatModel,
        vectorstore: VectorStore,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": top_k * 4, "lambda_mult": 0.7},
        )
        self._condense = CONDENSE_PROMPT | llm | StrOutputParser()
        self._answer = QA_PROMPT | llm | StrOutputParser()

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ) -> LegalRAG:
        from .models import get_embeddings, get_llm
        from .vectorstore import get_vectorstore

        settings = settings or get_settings()
        embeddings = embeddings or get_embeddings(settings)
        llm = llm or get_llm(settings)
        return cls(llm=llm, vectorstore=get_vectorstore(settings, embeddings), top_k=settings.top_k)

    # -- steps -------------------------------------------------------------------------

    def standalone_question(self, question: str, history: list[BaseMessage]) -> str:
        if not history:
            return question
        rewritten = self._condense.invoke({"history": history, "question": question}).strip()
        return rewritten or question

    def retrieve(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def _prepare(self, question: str, history: Sequence[dict[str, str]] | None):
        messages = to_messages(history)
        query = self.standalone_question(question, messages)
        docs = self.retrieve(query)
        inputs = {"context": format_context(docs), "history": messages, "question": question}
        return query, docs, inputs

    # -- public API --------------------------------------------------------------------

    def ask(self, question: str, history: Sequence[dict[str, str]] | None = None) -> dict[str, Any]:
        query, docs, inputs = self._prepare(question, history)
        answer = self._answer.invoke(inputs)
        return {
            "answer": answer,
            "standalone_question": query,
            "sources": [s.to_dict() for s in to_sources(docs)],
            "notices": detect_urgency(question),
            "disclaimer": DISCLAIMER,
        }

    def stream(
        self, question: str, history: Sequence[dict[str, str]] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield events: one ``meta`` event (sources, notices) then ``token`` events, then ``done``."""
        query, docs, inputs = self._prepare(question, history)
        yield {
            "type": "meta",
            "standalone_question": query,
            "sources": [s.to_dict() for s in to_sources(docs)],
            "notices": detect_urgency(question),
            "disclaimer": DISCLAIMER,
        }
        for token in self._answer.stream(inputs):
            if token:
                yield {"type": "token", "content": token}
        yield {"type": "done"}
