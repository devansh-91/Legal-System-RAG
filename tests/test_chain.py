import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from legal_rag.chain import LegalRAG, format_context, to_messages
from legal_rag.ingest import ingest
from legal_rag.safety import detect_urgency


@pytest.fixture
def rag_factory(settings, embeddings):
    ingest(settings, embeddings)

    def make(responses):
        llm = FakeListChatModel(responses=responses)
        return LegalRAG.from_settings(settings, llm=llm, embeddings=embeddings), llm

    return make


def test_ask_returns_answer_and_sources(rag_factory):
    rag, _ = rag_factory(["File a first appeal within 30 days [1]."])
    result = rag.ask("How do I file an RTI appeal?")
    assert result["answer"] == "File a first appeal within 30 days [1]."
    assert result["standalone_question"] == "How do I file an RTI appeal?"
    assert 1 <= len(result["sources"]) <= 3
    assert [s["id"] for s in result["sources"]] == list(range(1, len(result["sources"]) + 1))
    assert all(s["title"] and s["snippet"] for s in result["sources"])
    assert "not legal advice" in result["disclaimer"]


def test_history_triggers_condense_step(rag_factory):
    rag, llm = rag_factory(["What is the time limit for an RTI first appeal?", "30 days [1]."])
    history = [
        {"role": "user", "content": "Tell me about RTI"},
        {"role": "assistant", "content": "RTI lets you ask for information."},
    ]
    result = rag.ask("what about the appeal time?", history)
    assert result["standalone_question"] == "What is the time limit for an RTI first appeal?"
    assert result["answer"] == "30 days [1]."


def test_stream_events(rag_factory):
    rag, _ = rag_factory(["Call 1930."])
    events = list(rag.stream("Money was stolen in an online fraud"))
    assert events[0]["type"] == "meta"
    assert events[0]["notices"], "cyber fraud should surface the 1930 helpline"
    assert events[-1] == {"type": "done"}
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == "Call 1930."


def test_format_context_numbering():
    from langchain_core.documents import Document

    ctx = format_context([Document(page_content="a", metadata={"title": "T1"}),
                          Document(page_content="b", metadata={"title": "T2", "page": 4})])
    assert ctx.startswith("[1] (T1)\na")
    assert "[2] (T2, page 4)\nb" in ctx
    assert format_context([]) == "(no relevant passages found)"


def test_to_messages_truncates_and_filters():
    history = [{"role": "user", "content": str(i)} for i in range(10)] + [{"role": "system", "content": "x"}]
    msgs = to_messages(history)
    assert len(msgs) == 5  # last 6 turns, minus the unknown role


@pytest.mark.parametrize(
    "text,expected",
    [
        ("My husband is beating me every day", "112"),
        ("mujhe aatmahatya ke vichar aate hain आत्महत्या", "14416"),
        ("I was scammed via UPI fraud", "1930"),
        ("child marriage is happening next door", "1098"),
    ],
)
def test_detect_urgency(text, expected):
    notices = detect_urgency(text)
    assert any(expected in n for n in notices)


def test_detect_urgency_none():
    assert detect_urgency("How do I file an RTI?") == []
