# Data directory

Everything under this folder is indexed by `python -m legal_rag.ingest`.

- `knowledge_base/` — curated plain-language guides to common Indian legal topics (Markdown with a small
  front-matter block: `title`, `category`, `source`).
- `pdfs/` — drop official PDFs here (bare acts from https://www.indiacode.nic.in, judgments, government
  FAQs). Each page is indexed with its page number so answers can cite it.

After adding or editing files, re-run:

```bash
python -m legal_rag.ingest          # upsert new/changed chunks
python -m legal_rag.ingest --reset  # rebuild the index from scratch
```

The curated guides summarise the law as understood in 2025–26 and are meant for general
information. Always verify against the official text before relying on them.
