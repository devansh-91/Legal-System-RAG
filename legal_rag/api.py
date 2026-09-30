"""FastAPI backend + static web UI for the legal help site.

Run with:  uvicorn legal_rag.api:app --reload
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .chain import LegalRAG
from .config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)
WEB_DIR = PROJECT_ROOT / "web"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class SourceOut(BaseModel):
    id: int
    title: str
    section: str
    source: str
    reference: str
    snippet: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    standalone_question: str
    sources: list[SourceOut]
    notices: list[str]
    disclaimer: str


@lru_cache
def get_rag() -> LegalRAG:
    return LegalRAG.from_settings(get_settings())


def create_app() -> FastAPI:
    app = FastAPI(
        title="Nyaya Sahayak — Indian Legal Help (RAG)",
        version=__version__,
        description="Ask questions about Indian law. Answers are grounded in a curated "
        "knowledge base using LangChain + ChromaDB + free LLMs.",
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        settings = get_settings()
        return {"status": "ok", "llm_provider": settings.llm_provider, "version": __version__}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, rag: LegalRAG = Depends(get_rag)) -> dict:
        try:
            return rag.ask(req.question, [t.model_dump() for t in req.history])
        except Exception as exc:  # surface provider errors (bad key, model down) cleanly
            logger.exception("chat failed")
            raise HTTPException(status_code=502, detail=f"LLM/retrieval error: {exc}") from exc

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest, rag: LegalRAG = Depends(get_rag)) -> StreamingResponse:
        history = [t.model_dump() for t in req.history]

        def events():
            try:
                for event in rag.stream(req.question, history):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                logger.exception("stream failed")
                err = {"type": "error", "message": f"LLM/retrieval error: {exc}"}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
