# Gutermann CLI Q&A Agent

A local RAG-based CLI question-answering agent for the Gutermann water leak detection product knowledge base.

The agent retrieves relevant product information from a local knowledge base and generates grounded answers using a local Ollama language model.

## Current Architecture

- Markdown/product-aware knowledge base
- One chunk per product with product and category metadata
- Sentence Transformers embeddings
- ChromaDB local vector store
- BM25 keyword retrieval
- Hybrid retrieval combining semantic and lexical search
- Controlled `TOP_K` and retrieval candidate limits
- Query relevance checking
- Triage for greetings, questions, follow-ups, and out-of-scope requests
- Conversation memory
- Follow-up question resolution
- Grounded local LLM generation
- Ollama-based local inference

## Models

- LLM: `qwen2.5:3b`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

## Requirements

- Python 3.10+
- Ollama
- Qwen 2.5 3B model
- Python packages listed in `requirements.txt`

## Project Structure

```text
Gutermann-QA-Agent/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── follow_up.py
│   ├── generation.py
│   ├── ingestion.py
│   ├── memory.py
│   ├── models.py
│   ├── qa_agent.py
│   ├── relevance_checker.py
│   ├── retrieval.py
│   ├── triage.py
│   └── vector_store.py
│
├── tests/
│   ├── __init__.py
│   ├── test_generation.py
│   ├── test_ingestion.py
│   ├── test_memory.py
│   ├── test_problem_statement.py
│   ├── test_qa_agent.py
│   ├── test_triage.py
│   └── test_vector_store.py
│
├── product_overview.md
├── requirements.txt
├── README.md
└── .gitignore