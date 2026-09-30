from legal_rag.ingest import chunk_id, ingest, load_documents, parse_front_matter, split_documents
from legal_rag.models import get_embeddings  # noqa: F401  (import check only)
from legal_rag.vectorstore import get_vectorstore


def test_parse_front_matter():
    meta, body = parse_front_matter("---\ntitle: RTI\ncategory: Governance\n---\n# Heading\ntext")
    assert meta == {"title": "RTI", "category": "Governance"}
    assert body.startswith("# Heading")


def test_parse_front_matter_absent():
    meta, body = parse_front_matter("# Just markdown")
    assert meta == {}
    assert body == "# Just markdown"


def test_load_knowledge_base(settings):
    docs = load_documents(settings.data_dir)
    titles = {d.metadata["title"] for d in docs}
    assert len(docs) >= 10
    assert "Right to Information (RTI) Act, 2005" in titles
    # data/README.md must not be indexed
    assert all(not d.metadata["source"].lower().endswith("readme.md") for d in docs)


def test_split_keeps_section_metadata(settings):
    docs = load_documents(settings.data_dir)
    chunks = split_documents(docs, chunk_size=800, chunk_overlap=100)
    assert len(chunks) > len(docs)
    rti = [c for c in chunks if c.metadata["title"].startswith("Right to Information")]
    assert any("First appeal" in c.metadata["section"] for c in rti)
    assert all(c.page_content.startswith("[") for c in chunks)
    assert all(len(c.page_content) < 1000 for c in chunks)


def test_ingest_is_idempotent(settings, embeddings):
    first = ingest(settings, embeddings)
    second = ingest(settings, embeddings)
    assert first == second > 0
    store = get_vectorstore(settings, embeddings)
    assert len(store.get()["ids"]) == first


def test_ingest_reset(settings, embeddings):
    count = ingest(settings, embeddings)
    assert ingest(settings, embeddings, reset=True) == count


def test_chunk_id_stable(settings):
    docs = load_documents(settings.data_dir)
    chunks = split_documents(docs[:1], 800, 100)
    assert chunk_id(chunks[0]) == chunk_id(chunks[0])
    assert len({chunk_id(c) for c in chunks}) == len(chunks)
