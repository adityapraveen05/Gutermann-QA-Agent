# Gutermann CLI Q&A Agent

Local RAG-based product knowledge assistant for the technical assessment.

## Current architecture

- Markdown/product-aware ingestion
- One chunk per product with category/product/page metadata
- Local Ollama LLM
- Ollama embeddings
- ChromaDB local vector store
- BM25 keyword retrieval
- Hybrid retrieval
- Controlled TOP_K / FETCH_K
- MMR-style diversity selection
- Triage, memory, query rewriting and grounded generation modules

## Models

- LLM: `qwen2.5:3b`
- Embeddings: `nomic-embed-text`

## First setup

```powershell
ollama pull nomic-embed-text
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Test ingestion before running the full agent

```powershell
python test_ingestion.py
```

This should report 13 product chunks for the supplied knowledge base.

## Run the agent

```powershell
python qa_agent.py
```

Type `exit` or `quit` to end the session.
