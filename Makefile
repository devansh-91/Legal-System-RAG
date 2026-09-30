.PHONY: install ingest reset serve cli test

install:
	pip install -r requirements.txt

ingest:
	python -m legal_rag.ingest

reset:
	python -m legal_rag.ingest --reset

serve:
	uvicorn legal_rag.api:app --reload --port 8000

cli:
	python -m legal_rag.cli

test:
	pytest -q
