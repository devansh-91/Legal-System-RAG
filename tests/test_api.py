import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from legal_rag.api import create_app, get_rag
from legal_rag.chain import LegalRAG
from legal_rag.ingest import ingest


@pytest.fixture
def client(settings, embeddings):
    ingest(settings, embeddings)
    rag = LegalRAG.from_settings(
        settings, llm=FakeListChatModel(responses=["Answer [1]."]), embeddings=embeddings
    )
    app = create_app()
    app.dependency_overrides[get_rag] = lambda: rag
    return TestClient(app)


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Nyaya Sahayak" in res.text


def test_chat(client):
    res = client.post("/api/chat", json={"question": "How to file an FIR?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "Answer [1]."
    assert body["sources"]


def test_chat_validation(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    bad_history = {"question": "hi there", "history": [{"role": "system", "content": "x"}]}
    assert client.post("/api/chat", json=bad_history).status_code == 422


def test_chat_stream(client):
    res = client.post("/api/chat/stream", json={"question": "What is anticipatory bail?"})
    assert res.status_code == 200
    events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "done"
    assert "".join(e.get("content", "") for e in events) == "Answer [1]."
