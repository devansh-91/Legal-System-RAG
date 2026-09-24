"""Load legal documents from ``data/``, chunk them and index them in Chroma.

Supported inputs:
  * Markdown (``.md``) with an optional front-matter block (title/category/source)
  * Plain text (``.txt``)
  * PDF (``.pdf``) — e.g. bare acts downloaded from https://www.indiacode.nic.in

Usage::

    python -m legal_rag.ingest            # index everything under DATA_DIR
    python -m legal_rag.ingest --reset    # wipe the collection first
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a simple ``key: value`` front-matter block from the body."""
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end():]


def load_documents(data_dir: Path) -> list[Document]:
    """Load every supported file under ``data_dir`` into LangChain documents."""
    docs: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or path.name.lower() == "readme.md":
            continue
        suffix = path.suffix.lower()
        rel = path.relative_to(data_dir).as_posix()
        if suffix in (".md", ".txt"):
            meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
            docs.append(
                Document(
                    page_content=body,
                    metadata={
                        "source": rel,
                        "title": meta.get("title", path.stem.replace("_", " ").title()),
                        "category": meta.get("category", "General"),
                        "reference": meta.get("source", ""),
                        "format": suffix.lstrip("."),
                    },
                )
            )
        elif suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            for page in PyPDFLoader(str(path)).load():
                page.metadata = {
                    "source": rel,
                    "title": path.stem.replace("_", " ").title(),
                    "category": "Bare Act",
                    "reference": "",
                    "format": "pdf",
                    "page": int(page.metadata.get("page", 0)) + 1,
                }
                docs.append(page)
    logger.info("Loaded %d documents from %s", len(docs), data_dir)
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split markdown by headings first (to keep section context), then by size."""
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Document] = []
    for doc in docs:
        if doc.metadata.get("format") == "md":
            sections = header_splitter.split_text(doc.page_content)
            for section in sections:
                heading = " › ".join(
                    section.metadata[key] for key in ("h1", "h2", "h3") if key in section.metadata
                )
                meta = {**doc.metadata, "section": heading}
                for piece in size_splitter.split_text(section.page_content):
                    chunks.append(Document(page_content=piece, metadata=meta))
        else:
            for piece in size_splitter.split_documents([doc]):
                piece.metadata.setdefault("section", "")
                chunks.append(piece)

    # Prefix each chunk with its title/section so the embedding carries that context.
    for index, chunk in enumerate(chunks):
        title = chunk.metadata.get("title", "")
        section = chunk.metadata.get("section", "")
        header = f"{title} — {section}" if section else title
        if header and not chunk.page_content.startswith(header):
            chunk.page_content = f"[{header}]\n{chunk.page_content}"
        chunk.metadata["chunk"] = index
    return chunks


def chunk_id(chunk: Document) -> str:
    """Stable ID so re-running ingestion upserts instead of duplicating."""
    raw = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page', '')}|{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def ingest(
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    reset: bool = False,
) -> int:
    """Index all documents in ``settings.data_dir``. Returns the number of chunks."""
    from .models import get_embeddings
    from .vectorstore import get_vectorstore

    settings = settings or get_settings()
    embeddings = embeddings or get_embeddings(settings)
    store = get_vectorstore(settings, embeddings)
    if reset:
        store.reset_collection()

    docs = load_documents(settings.data_dir)
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        logger.warning("No documents found in %s", settings.data_dir)
        return 0

    ids = [chunk_id(c) for c in chunks]
    # Drop duplicate chunks (identical text in the same file) before upserting.
    unique = dict(zip(ids, chunks))
    batch = 256
    items = list(unique.items())
    for start in range(0, len(items), batch):
        part = items[start : start + batch]
        store.add_documents([doc for _, doc in part], ids=[i for i, _ in part])
    logger.info("Indexed %d chunks into collection %r", len(unique), settings.chroma_collection)
    return len(unique)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index legal documents into Chroma")
    parser.add_argument("--reset", action="store_true", help="delete existing vectors first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count = ingest(reset=args.reset)
    print(f"Done. {count} chunks indexed.")


if __name__ == "__main__":
    main()
