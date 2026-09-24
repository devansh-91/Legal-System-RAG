# ⚖️ Nyaya Sahayak — Indian Legal Help RAG

A retrieval-augmented generation (RAG) assistant for an **Indian legal help website**. It answers
questions about Indian law in plain language (English, Hindi or Hinglish), **cites its sources**, and
points people to the right authority and helpline.

Built with **LangChain + ChromaDB + free models only**:

| Piece | What's used | Cost |
|---|---|---|
| Orchestration | LangChain (LCEL) | Free / open source |
| Vector DB | ChromaDB (persistent, local) | Free / open source |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers (runs on CPU) | Free, local |
| LLM | **Groq** free tier (Llama 3.3 70B), **Ollama** (fully local), or **Hugging Face** Inference | Free |
| Backend | FastAPI (+ server-sent-event streaming) | Free |
| Frontend | Plain HTML/CSS/JS chat UI, mobile friendly, dark mode | Free |

## Features

- **Curated knowledge base** (`data/knowledge_base/`) covering FIRs, arrest & bail, the new
  **BNS/BNSS/BSA** laws (with an IPC → BNS section map), consumer complaints, RTI, domestic violence,
  maintenance & divorce, cheque bounce, tenancy & property, cyber fraud, free legal aid & Lok Adalats,
  fundamental rights & writs, workplace rights (POSH, gratuity), motor accidents, senior citizens and
  children.
- **Drop in your own PDFs** (bare acts from [India Code](https://www.indiacode.nic.in), judgments,
  FAQs) into `data/pdfs/` — they're chunked and indexed with page numbers.
- **Conversational memory** — follow-up questions are rewritten into standalone queries before retrieval.
- **Grounded answers with inline citations** `[1]`, plus a sources panel with snippets.
- **MMR retrieval** for diverse context; heading-aware chunking so each chunk knows its section.
- **Urgency detection** — mentions of violence, self-harm, child abuse or cyber fraud surface the
  right helpline (112, 181, 1098, 1930, 14416) before the answer.
- **Idempotent ingestion** — stable chunk IDs, so re-indexing upserts instead of duplicating.
- Streaming chat UI, REST API, and a terminal CLI.

## Quick start

```bash
git clone https://github.com/devansh-91/Legal-System-RAG.git
cd Legal-System-RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env         # then pick an LLM provider (see below)
python -m legal_rag.ingest   # builds ./chroma_db (downloads the embedding model the first time)
uvicorn legal_rag.api:app --reload
```

Open http://localhost:8000 for the chat site, or http://localhost:8000/docs for the API docs.

Terminal chat: `python -m legal_rag.cli`

### Choosing a free LLM

| Provider | Setup | Notes |
|---|---|---|
| `groq` *(default)* | Free key from https://console.groq.com/keys → `GROQ_API_KEY=...` | Fastest; generous free tier |
| `ollama` | Install https://ollama.com, run `ollama pull llama3.2` → `LLM_PROVIDER=ollama` | 100% local & private, no key |
| `huggingface` | Token from https://huggingface.co/settings/tokens → `HUGGINGFACEHUB_API_TOKEN=...`, `LLM_PROVIDER=huggingface` | Uses HF's free inference credits |

For better Hindi/regional-language retrieval set `EMBEDDING_MODEL=intfloat/multilingual-e5-small`
and re-run `python -m legal_rag.ingest --reset` (changing the embedding model requires a fresh index).

### Docker

```bash
cp .env.example .env   # add your GROQ_API_KEY
docker compose up --build
# or, fully local with Ollama:
#   set LLM_PROVIDER=ollama in .env
#   docker compose --profile local up --build
#   docker compose exec ollama ollama pull llama3.2
```

## API

`POST /api/chat`

```json
{
  "question": "What if I get no reply to my RTI?",
  "history": [
    {"role": "user", "content": "How do I file an RTI?"},
    {"role": "assistant", "content": "..."}
  ]
}
```

Response:

```json
{
  "answer": "If you don't get a reply within 30 days, it is treated as a refusal ... [1]",
  "standalone_question": "What can I do if I get no reply to my RTI application?",
  "sources": [{"id": 1, "title": "Right to Information (RTI) Act, 2005", "section": "...", "source": "knowledge_base/05_rti.md", "reference": "...", "snippet": "...", "page": null}],
  "notices": [],
  "disclaimer": "This is general legal information, not legal advice. ..."
}
```

`POST /api/chat/stream` takes the same body and returns server-sent events: one `meta` event
(sources, notices), then `token` events, then `done` (or `error`).

`GET /api/health` — liveness + configured provider.

## How it works

```
question + chat history
        │
        ▼
 condense (LLM) ──► standalone question
        │
        ▼
 Chroma MMR retrieval (bge-small embeddings, top-k chunks)
        │
        ▼
 numbered context [1]..[k] + system rules (cite, no invented sections,
 BNS/BNSS awareness, reply in user's language, emergency → 112)
        │
        ▼
 LLM (Groq / Ollama / HF) ──► answer with citations + sources + helpline notices
```

## Project layout

```
legal_rag/
  config.py       settings from env / .env
  models.py       free LLM + embedding factories (groq / ollama / huggingface)
  vectorstore.py  Chroma setup
  ingest.py       load .md/.txt/.pdf → heading-aware chunks → Chroma
  prompts.py      system + condense prompts, disclaimer
  safety.py       urgency detection → helplines
  chain.py        the RAG pipeline (ask + stream)
  api.py          FastAPI app + static site
  cli.py          terminal chat
web/              chat UI (index.html, style.css, app.js)
data/
  knowledge_base/ curated Markdown guides
  pdfs/           put official PDFs here
tests/            pytest suite (uses fake LLM + fake embeddings, runs offline)
```

## Adding knowledge

Add Markdown files to `data/knowledge_base/` with a front-matter block:

```markdown
---
title: Name of the topic
category: Family Law
source: Act name and sections
---

# Name of the topic
## Sub-heading
Plain-language explanation…
```

or drop PDFs into `data/pdfs/`, then run `python -m legal_rag.ingest`.

## Tests

```bash
pytest -q
```

Tests use LangChain's fake chat model and deterministic fake embeddings, so they need no API keys or
model downloads.

## Disclaimer

This project provides **general legal information, not legal advice**. The curated guides summarise
Indian law as understood in 2025–26 and may be incomplete or out of date; laws, section numbers and
amounts change. Always verify against the official text (https://www.indiacode.nic.in) and consult an
advocate or your District Legal Services Authority (helpline **15100**) for your specific situation.
